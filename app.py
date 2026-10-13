import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')

# Import segmentation pipeline
from segmentation import AdvancedCustomerSegmentation

# 1. Page Configuration
st.set_page_config(page_title="Advanced Customer Analytics", layout="wide")
st.title("Customer Intelligence Platform")

# Initialize session state
if 'segments_df' not in st.session_state:
    st.session_state.segments_df = None
if 'tx_df' not in st.session_state:
    st.session_state.tx_df = None
if 'segments_generated' not in st.session_state:
    st.session_state.segments_generated = False

# 2. File Upload (for Streamlit Cloud deployment) - ONLY TRANSACTIONS NEEDED
st.sidebar.header("📁 Data Upload")
st.sidebar.markdown("""
Upload **cleaned transaction data** only. The app will generate segments on demand.
""")

uploaded_transactions = st.sidebar.file_uploader(
    "Transactions Parquet (customer_transactions_clean.parquet)",
    type=["parquet"],
    help="Cleaned transaction data - the app will generate segments automatically"
)

# Load transactions (with local fallback)
@st.cache_data
def load_transactions_from_file(uploaded_file):
    """Load transactions from uploaded file or local fallback."""
    if uploaded_file is not None:
        return pl.read_parquet(io.BytesIO(uploaded_file.read()))
    # Fallback to local file
    try:
        return pl.read_parquet("customer_transactions_clean.parquet")
    except FileNotFoundError:
        st.warning("No transactions file found. Please upload a parquet file.")
        return None

# Load or generate segments
def get_or_generate_segments(tx_df):
    """Load existing segments or generate new ones from transactions."""
    # Try to load local segments file first
    try:
        segments = pl.read_parquet("customer_segments_advanced.parquet")
        st.sidebar.success("✅ Loaded existing segments from local file")
        return segments
    except FileNotFoundError:
        pass
    
    # If no local file, run pipeline on provided transactions
    if tx_df is not None:
        with st.spinner("Running segmentation pipeline (Winsorization → RobustScaler → PCA → KMeans)... This takes ~30-60 seconds."):
            pipeline = AdvancedCustomerSegmentation(max_clusters=8, n_components=6)
            segments = pipeline.fit_predict(tx_df)
            # Save to local file for future runs
            segments.write_parquet("customer_segments_advanced.parquet")
            st.sidebar.success("✅ Segments generated and saved!")
            return segments
    
    st.error("No transaction data available. Please upload a parquet file.")
    st.stop()

@st.cache_data
def generate_retention_matrix_polars(transactions_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Generates a Monthly Repeat-Purchase Retention matrix entirely in Polars.
    Returns a pivoted DataFrame with cohorts as rows, periods as columns, retention rates as values.
    """
    df = transactions_lf.with_columns([
        pl.col('transaction_date').dt.truncate('1mo').alias('OrderMonth'),
    ]).with_columns([
        pl.col('transaction_date').dt.truncate('1mo').alias('tx_month'),
    ])
    
    # Get cohort month (first purchase month per customer)
    cohort_months = df.group_by('customer_id').agg(
        pl.col('tx_month').min().alias('CohortMonth')
    )
    
    df = df.join(cohort_months, on='customer_id', how='left')
    
    # Calculate period number (months since cohort)
    df = df.with_columns([
        ((pl.col('OrderMonth').dt.year() - pl.col('CohortMonth').dt.year()) * 12 + 
         (pl.col('OrderMonth').dt.month() - pl.col('CohortMonth').dt.month())).alias('PeriodNumber')
    ])
    
    # Count unique customers per cohort/period
    cohort_data = df.group_by(['CohortMonth', 'PeriodNumber']).agg(
        pl.col('customer_id').n_unique().alias('customer_count')
    ).collect()
    
    # Pivot to matrix
    cohort_pivot = cohort_data.pivot(
        values='customer_count',
        index='CohortMonth',
        on='PeriodNumber',
        aggregate_function='first'
    ).sort('CohortMonth')
    
    # Fill nulls with 0 for completed periods
    cohort_pivot = cohort_pivot.fill_null(0)
    
    # Normalize each period by its cohort's Month 0 customer count
    period_cols = [c for c in cohort_pivot.columns if c != 'CohortMonth' and isinstance(c, (int, str))]
    period_cols_int = []
    for c in period_cols:
        try:
            period_cols_int.append(int(c))
        except (ValueError, TypeError):
            pass
    
    if period_cols_int:
        period_cols_int.sort()
        # Get Month 0 cohort sizes (column names are strings in Polars)
        if 0 in period_cols_int:
            cohort_sizes = cohort_pivot.select(pl.col("0")).to_series()
            for pc in period_cols_int:
                pc_str = str(pc)
                if pc != 0:
                    cohort_pivot = cohort_pivot.with_columns(
                        (pl.col(pc_str) / cohort_sizes).alias(pc_str)
                    )
            # Month 0 is always 100%
            cohort_pivot = cohort_pivot.with_columns(pl.lit(1.0).alias("0"))
    
    # Convert CohortMonth to string for display
    cohort_pivot = cohort_pivot.with_columns(
        pl.col('CohortMonth').dt.strftime('%Y-%m').alias('Cohort')
    ).drop('CohortMonth')
    
    # Ensure period columns are sorted numerically
    new_cols = ['Cohort'] + [str(pc) for pc in period_cols_int]
    return cohort_pivot.select(new_cols)

# ============================================================
# MAIN DATA LOADING LOGIC
# ============================================================

# Load transactions
if uploaded_transactions is not None or st.session_state.tx_df is not None:
    if uploaded_transactions is not None:
        # New upload - load it
        tx_df = pl.read_parquet(io.BytesIO(uploaded_transactions.read()))
        st.session_state.tx_df = tx_df
        st.session_state.segments_generated = False  # Reset segments flag
    else:
        # Use session state
        tx_df = st.session_state.tx_df
else:
    # Try local file
    try:
        tx_df = pl.read_parquet("customer_transactions_clean.parquet")
        st.session_state.tx_df = tx_df
    except FileNotFoundError:
        st.error("No transaction data available. Please upload a parquet file.")
        st.stop()

tx_lazy = tx_df.lazy()

# Generate or load segments
if not st.session_state.segments_generated or st.session_state.segments_df is None:
    # Show generate button
    if st.sidebar.button("🔄 Generate / Refresh Segments", type="primary"):
        segments_df = get_or_generate_segments(tx_df)
        st.session_state.segments_df = segments_df
        st.session_state.segments_generated = True
        st.rerun()
    else:
        st.info("👈 Click **'Generate / Refresh Segments'** in the sidebar to run the segmentation pipeline.")
        st.stop()
else:
    segments_df = st.session_state.segments_df

# Add Primary_Category_Name if missing (derive from transactions in Polars)
if 'Primary_Category_Name' not in segments_df.columns:
    primary_cat = tx_lazy.group_by('customer_id').agg(
        pl.col('category').mode().first().alias('Primary_Category_Name')
    ).collect()
    primary_cat = primary_cat.with_columns(
        pl.col('Primary_Category_Name').fill_null('Unknown')
    )
    segments_df = segments_df.join(primary_cat, on='customer_id', how='left')
    segments_df = segments_df.with_columns(
        pl.col('Primary_Category_Name').fill_null('Unknown')
    )
    # Update session state
    st.session_state.segments_df = segments_df

# ============================================================
# SHARED CLUSTER COLOR MAPPING & HELPERS
# ============================================================
clusters = sorted(segments_df['Cluster'].unique().to_list())
cluster_labels = [f"Cluster {c}" for c in clusters]
palette = px.colors.qualitative.Safe

CLUSTER_COLORS = {
    label: palette[i % len(palette)]
    for i, label in enumerate(cluster_labels)
}

CLUSTER_CATEGORY_ORDERS = {"Segment": cluster_labels}

def add_segment_column(df: pl.DataFrame) -> pl.DataFrame:
    """Add human-readable Segment column for consistent coloring."""
    return df.with_columns(
        pl.col('Cluster').cast(pl.Utf8).map_elements(
            lambda x: f"Cluster {x}", return_dtype=pl.Utf8
        ).alias('Segment')
    )

def filter_empty_warning(filtered_df: pl.DataFrame) -> bool:
    """Show warning and stop if no clusters selected."""
    if filtered_df.is_empty():
        st.info("Select at least one cluster to explore customer behavior.")
        st.stop()
        return True
    return False

# ============================================================
# UI LAYOUT
# ============================================================

# Sidebar Filters
st.sidebar.header("🎯 Global Filters")
selected_clusters = st.sidebar.multiselect(
    "Select Clusters", 
    clusters, 
    default=clusters,
    format_func=lambda x: f"Cluster {x}"
)
filtered_df = segments_df.filter(pl.col('Cluster').is_in(selected_clusters))

# Handle empty selection
filter_empty_warning(filtered_df)

# Add Segment column for plotting
filtered_df = add_segment_column(filtered_df)
segments_df = add_segment_column(segments_df)

# ============================================================
# SUMMARY CARDS
# ============================================================
total_customers = segments_df.height
selected_count = filtered_df.height
selected_share = selected_count / total_customers * 100
selected_spend = filtered_df.select(pl.col('Total_Lifetime_Spend').sum()).item()
median_aov = filtered_df.select(pl.col('Average_Order_Value_AOV').median()).item()
median_recency = filtered_df.select(pl.col('Recency_Days').median()).item()

# Dataset metadata
max_date = tx_lazy.select(pl.col('transaction_date').max()).collect().item()
min_date = tx_lazy.select(pl.col('transaction_date').min()).collect().item()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Selected Customers", f"{selected_count:,}", f"{selected_share:.1f}% of base")
with col2:
    st.metric("Selected Spend (Observed)", f"${selected_spend:,.0f}")
with col3:
    st.metric("Median AOV", f"${median_aov:.2f}")
with col4:
    st.metric("Median Recency (Days)", f"{median_recency:.0f}")

st.caption(f"Dataset range: {min_date.date()} to {max_date.date()} | Recency reference: {max_date.date() + pl.duration(days=1)}")

# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Risk–Value Portfolio", 
    "Segment Profiles", 
    "Distributions", 
    "Model Exploration"
])

# ------------------------------------------------------------------
# TAB 1: RISK–VALUE PORTFOLIO (Business-facing scatter)
# ------------------------------------------------------------------
with tab1:
    st.header("Risk–Value Portfolio")
    st.markdown("""
    **Which valuable customers are overdue for a purchase?**
    
    - **X-axis**: Purchase-Delay Index = Days Since Last Purchase ÷ Avg Days Between Purchases
    - **Y-axis**: Spend in Observed History (log scale available)
    - **Color**: Segment (consistent across all views)
    - **Hover**: Customer ID, Spend, Recency, Avg Purchase Interval, AOV
    
    *Customers with insufficient repeat-purchase history (index = 0) are shown separately.*
    """)
    
    # Prepare plot data
    plot_df = filtered_df.select([
        'customer_id', 'Cluster', 'Segment', 'Total_Lifetime_Spend', 
        'Churn_Risk_Index_CRI', 'Recency_Days', 
        'Average_InterPurchase_Time_AIPT', 'Average_Order_Value_AOV'
    ]).to_pandas()
    
    # Separate customers with sufficient history
    has_history = plot_df['Average_InterPurchase_Time_AIPT'] > 0
    no_history = plot_df[~has_history]
    plot_df = plot_df[has_history]
    
    col1, col2 = st.columns([3, 1])
    with col2:
        log_scale = st.checkbox("Log scale (Y-axis)", value=True)
        show_no_history = st.checkbox("Show 'Insufficient history'", value=True)
    
    fig_scatter = px.scatter(
        plot_df,
        x='Churn_Risk_Index_CRI',
        y='Total_Lifetime_Spend',
        color='Segment',
        color_discrete_map=CLUSTER_COLORS,
        category_orders=CLUSTER_CATEGORY_ORDERS,
        hover_data=['customer_id', 'Recency_Days', 'Average_InterPurchase_Time_AIPT', 'Average_Order_Value_AOV'],
        title="High Value / High Delay Matrix",
        log_y=log_scale,
        size_max=15,
    )
    
    # Add reference lines for illustrative zones
    fig_scatter.add_vline(x=1.0, line_dash="dot", line_color="gray", 
                          annotation_text="On Cadence", annotation_position="top left")
    fig_scatter.add_vline(x=2.0, line_dash="dot", line_color="orange", 
                          annotation_text="2× Cadence", annotation_position="top left")
    fig_scatter.add_vline(x=3.0, line_dash="dash", line_color="red", 
                          annotation_text="Critical Delay", annotation_position="top right")
    
    # Add no-history points at x=0 with different symbol
    if show_no_history and not no_history.empty:
        fig_scatter.add_scatter(
            x=no_history['Churn_Risk_Index_CRI'],
            y=no_history['Total_Lifetime_Spend'],
            mode='markers',
            marker=dict(symbol='x', size=8, color='lightgray'),
            name='Insufficient history',
            showlegend=True,
            hovertext=no_history['customer_id'].astype(str) + ': ' + no_history['Total_Lifetime_Spend'].astype(str)
        )
    
    fig_scatter.update_layout(
        xaxis_title="Purchase-Delay Index (Recency ÷ Avg Inter-Purchase Days)",
        yaxis_title="Spend in Observed History ($)",
        height=600,
        legend_title="Segment"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Quick actionable table
    st.subheader("Top Targets: High Spend + High Delay")
    high_delay = plot_df[plot_df['Churn_Risk_Index_CRI'] >= 2.0].nlargest(20, 'Total_Lifetime_Spend')
    if not high_delay.empty:
        st.dataframe(high_delay[['customer_id', 'Segment', 'Total_Lifetime_Spend', 
                                  'Churn_Risk_Index_CRI', 'Recency_Days', 'Average_Order_Value_AOV']])
    else:
        st.info("No customers with delay index ≥ 2.0 in selected segments.")

# ------------------------------------------------------------------
# TAB 2: SEGMENT PROFILES (Benchmark Heatmap)
# ------------------------------------------------------------------
with tab2:
    st.header("Segment Profiles")
    st.markdown("""
    **How does each segment differ from the overall customer base?**
    
    Colors show standardized differences (segment mean − overall mean) ÷ overall std.
    Red = above average, Blue = below average. White = near overall average.
    """)
    
    # Define metrics with readable labels
    metric_config = [
        ('Total_Lifetime_Spend', 'Spend in Observed History ($)'),
        ('Average_Order_Value_AOV', 'Average Order Value ($)'),
        ('Recency_Days', 'Days Since Last Purchase'),
        ('Total_Lifetime_Transactions', 'Total Transactions'),
        ('Category_Width', 'Categories Purchased'),
        ('Average_InterPurchase_Time_AIPT', 'Avg Days Between Purchases'),
        ('Churn_Risk_Index_CRI', 'Purchase-Delay Index'),
        ('Inferred_Discount_Affinity', 'Discount Affinity (ratio)'),
    ]
    
    available_metrics = [m for m, _ in metric_config if m in filtered_df.columns]
    metric_labels = {m: label for m, label in metric_config if m in available_metrics}
    
    if available_metrics:
        # Compute overall benchmarks from FULL dataset (not filtered)
        overall_means = segments_df.select([pl.col(m).mean().alias(m) for m in available_metrics]).to_pandas().iloc[0]
        overall_stds = segments_df.select([pl.col(m).std().alias(m) for m in available_metrics]).to_pandas().iloc[0]
        
        # Compute segment means
        segment_means = filtered_df.group_by('Segment').agg([
            pl.col(m).mean().alias(m) for m in available_metrics
        ]).to_pandas()
        
        # Standardize: (segment_mean - overall_mean) / overall_std
        z_data = segment_means.copy()
        for m in available_metrics:
            mean_val = overall_means[m]
            std_val = overall_stds[m]
            if std_val > 0:
                z_data[m] = (z_data[m] - mean_val) / std_val
            else:
                z_data[m] = 0
        
        # Prepare hover text with raw values
        hover_text = []
        for _, row in segment_means.iterrows():
            segment = row['Segment']
            texts = []
            for m in available_metrics:
                raw = row[m]
                overall = overall_means[m]
                texts.append(f"{metric_labels[m]}: {raw:.2f} (overall: {overall:.2f})")
            hover_text.append("<br>".join(texts))
        
        z_values = z_data[available_metrics].values
        
        fig_heatmap = px.imshow(
            z_values,
            x=[metric_labels[m] for m in available_metrics],
            y=z_data['Segment'].tolist(),
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0,
            text_auto=".2f",
            aspect="auto",
            title="Standardized Segment Differences from Customer Base",
        )
        fig_heatmap.update_layout(
            xaxis_title="",
            yaxis_title="",
            height=400 + len(z_data) * 30,
            coloraxis_colorbar=dict(title="Std. Diff."),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.caption("""
        **How to read:** Values are (segment mean − overall mean) / overall std. 
        A value of +1.0 means the segment averages one standard deviation above the customer base.
        Colors indicate direction and magnitude of difference, not whether a segment is 'good' or 'bad'.
        """)
        
        # Optional Radar Chart (expander)
        with st.expander("Radar View (min-max normalized within selection)"):
            st.markdown("*Shape depends on filtered selection; use heatmap above for stable comparison.*")
            
            # Min-max normalize within filtered selection
            radar_df = segment_means.copy()
            for m in available_metrics:
                max_val = radar_df[m].max()
                min_val = radar_df[m].min()
                if max_val > min_val:
                    radar_df[m] = (radar_df[m] - min_val) / (max_val - min_val)
                else:
                    radar_df[m] = 0.5
            
            fig_radar = go.Figure()
            for _, row in radar_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=row[available_metrics].values,
                    theta=[metric_labels[m] for m in available_metrics],
                    fill='toself',
                    name=row['Segment'],
                    line=dict(color=CLUSTER_COLORS.get(row['Segment'], 'gray'))
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=500,
                title="Min-Max Normalized Profiles (Within Selection)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("Required metrics not found in segment data.")

# ------------------------------------------------------------------
# TAB 3: DISTRIBUTIONS (Box Plots)
# ------------------------------------------------------------------
with tab3:
    st.header("Distributions Within Segments")
    st.markdown("""
    Cluster averages hide variation. Select a metric to see the full distribution per segment.
    """)
    
    dist_metrics = [
        ('Total_Lifetime_Spend', 'Spend in Observed History ($)', True),
        ('Average_Order_Value_AOV', 'Average Order Value ($)', True),
        ('Recency_Days', 'Days Since Last Purchase', False),
        ('Total_Lifetime_Transactions', 'Total Transactions', True),
        ('Churn_Risk_Index_CRI', 'Purchase-Delay Index', False),
        ('Category_Width', 'Categories Purchased', False),
        ('Average_InterPurchase_Time_AIPT', 'Avg Days Between Purchases', False),
    ]
    
    available_dist = [(m, label, log) for m, label, log in dist_metrics if m in filtered_df.columns]
    
    if available_dist:
        col1, col2 = st.columns([3, 1])
        with col2:
            selected_metric, selected_label, default_log = st.selectbox(
                "Explore distribution:",
                options=[m for m, _, _ in available_dist],
                format_func=lambda x: dict(available_dist)[x],
                index=0
            )
            log_scale = st.checkbox("Log scale (Y-axis)", value=default_log)
        
        plot_df = filtered_df.select(['Segment', selected_metric]).to_pandas()
        
        fig_box = px.box(
            plot_df,
            x='Segment',
            y=selected_metric,
            color='Segment',
            color_discrete_map=CLUSTER_COLORS,
            category_orders=CLUSTER_CATEGORY_ORDERS,
            title=f"Distribution of {selected_label} by Segment",
            points='outliers',
            log_y=log_scale,
        )
        fig_box.update_layout(
            xaxis_title="Segment",
            yaxis_title=selected_label,
            height=500,
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Summary stats
        stats = filtered_df.group_by('Segment').agg([
            pl.col(selected_metric).median().alias('median'),
            pl.col(selected_metric).mean().alias('mean'),
            pl.col(selected_metric).std().alias('std'),
            pl.len().alias('count'),
        ]).to_pandas()
        stats = stats.rename(columns={'median': 'Median', 'mean': 'Mean', 'std': 'Std Dev', 'count': 'Customers'})
        st.dataframe(stats, hide_index=True)
    else:
        st.warning("No distribution metrics available.")

# ------------------------------------------------------------------
# TAB 4: MODEL EXPLORATION (PCA)
# ------------------------------------------------------------------
with tab4:
    st.header("Model Exploration: PCA Projection")
    st.markdown("""
    Diagnostic view of the clustering in principal component space.
    
    - **Components**: First 3 principal components from robust PCA (6 components, 85% variance)
    - **Explained variance**: Displayed below
    - *Note: 2D/3D projections show partial views; overlap here doesn't mean overlap in full space.*
    """)
    
    pca_cols = [c for c in segments_df.columns if c.startswith('PC')]
    if len(pca_cols) >= 3:
        col1, col2 = st.columns([3, 1])
        with col2:
            view = st.radio("Projection", ["PC1 vs PC2", "PC1 vs PC3", "3D (PC1/PC2/PC3)"])
            show_explained = st.checkbox("Show explained variance", value=True)
        
        plot_df = filtered_df.select([
            'customer_id', 'Cluster', 'Segment', 'Total_Lifetime_Spend',
            'Average_Order_Value_AOV', 'Primary_Category_Name',
            'PC1', 'PC2', 'PC3'
        ]).to_pandas()
        
        if view == "3D (PC1/PC2/PC3)":
            fig_pca = px.scatter_3d(
                plot_df, x='PC1', y='PC2', z='PC3',
                color='Segment',
                color_discrete_map=CLUSTER_COLORS,
                category_orders=CLUSTER_CATEGORY_ORDERS,
                hover_data=['customer_id', 'Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Primary_Category_Name'],
                size='Total_Lifetime_Spend',
                size_max=15,
                opacity=0.7,
                title="3D Behavioral PCA Space"
            )
            fig_pca.update_layout(
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'
                ),
                height=700
            )
        elif view == "PC1 vs PC3":
            fig_pca = px.scatter(
                plot_df, x='PC1', y='PC3',
                color='Segment',
                color_discrete_map=CLUSTER_COLORS,
                category_orders=CLUSTER_CATEGORY_ORDERS,
                hover_data=['customer_id', 'Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Primary_Category_Name'],
                title="PCA: PC1 vs PC3"
            )
            fig_pca.update_layout(height=600)
        else:  # PC1 vs PC2
            fig_pca = px.scatter(
                plot_df, x='PC1', y='PC2',
                color='Segment',
                color_discrete_map=CLUSTER_COLORS,
                category_orders=CLUSTER_CATEGORY_ORDERS,
                hover_data=['customer_id', 'Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Primary_Category_Name'],
                title="PCA: PC1 vs PC2"
            )
            fig_pca.update_layout(height=600)
        
        st.plotly_chart(fig_pca, use_container_width=True)
        
        if show_explained:
            st.info("""
            **PCA Explained Variance (from training):**
            - PC1: ~35-40% (Spend & Volume axis)
            - PC2: ~20-25% (Cadence & Frequency axis)  
            - PC3: ~15-20% (Category Depth axis)
            - Total (6 components): ~85%
            
            *Loadings not currently exported; axis labels are interpretive.*
            """)
    else:
        st.warning("PCA components (PC1, PC2, PC3) not found in segment data. Re-run segmentation to generate them.")

# ------------------------------------------------------------------
# COHORT RETENTION (Sidebar toggle to avoid heavy compute)
# ------------------------------------------------------------------
st.sidebar.markdown("---")
if tx_lazy is None:
    st.sidebar.warning("⚠️ No transactions data loaded. Cohort retention unavailable.")
else:
    if st.sidebar.button("Generate Monthly Repeat-Purchase Retention"):
        with st.spinner("Computing cohort retention..."):
            retention_df = generate_retention_matrix_polars(tx_lazy)
            
            if retention_df.height > 0 and retention_df.width > 1:
                retention_pd = retention_df.to_pandas()
                retention_pd = retention_pd.set_index('Cohort')
                
                fig_heatmap = px.imshow(
                    retention_pd.values,
                    x=retention_pd.columns.astype(str),
                    y=retention_pd.index,
                    text_auto=".0%",
                    color_continuous_scale="YlGnBu",
                    color_continuous_midpoint=0.5,
                    range_color=[0, 1],
                    title="Monthly Repeat-Purchase Retention",
                    aspect="auto"
                )
                fig_heatmap.update_layout(
                    xaxis_title="Months Since First Observed Purchase",
                    yaxis_title="First Observed Purchase Cohort",
                    height=600
                )
                st.sidebar.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Summary metrics
                period_cols = [c for c in retention_df.columns if c != 'Cohort']
                col1, col2, col3 = st.columns(3)
                with col1:
                    if '1' in period_cols:
                        m1 = retention_df.select(pl.col('1').mean()).item()
                        st.metric("Avg Month 1 Retention", f"{m1:.1%}")
                with col2:
                    if '3' in period_cols:
                        m3 = retention_df.select(pl.col('3').mean()).item()
                        st.metric("Avg Month 3 Retention", f"{m3:.1%}")
                with col3:
                    st.metric("Cohorts", retention_df.height)
            else:
                st.sidebar.warning("Insufficient transaction history for cohort analysis.")

# Segment Summary in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Segment Summary")
for c in clusters:
    subset = filtered_df.filter(pl.col('Cluster') == c)
    count = subset.height
    avg_spend = subset.select(pl.col('Total_Lifetime_Spend').mean()).item()
    avg_aov = subset.select(pl.col('Average_Order_Value_AOV').mean()).item()
    churn_risk = subset.select(pl.col('Churn_Risk_Index_CRI').mean()).item()
    top_cat = subset.select(pl.col('Primary_Category_Name').mode().first()).item()
    
    st.sidebar.markdown(f"""
    **Cluster {c}** ({count} customers)
    - Avg Spend: ${avg_spend:,.0f}
    - Avg AOV: ${avg_aov:.2f}
    - Delay Index: {churn_risk:.2f}
    - Top Category: {top_cat}
    """)