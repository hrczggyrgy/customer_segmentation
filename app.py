import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import hashlib
import json
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
if 'tx_fingerprint' not in st.session_state:
    st.session_state.tx_fingerprint = None
if 'segments_fingerprint' not in st.session_state:
    st.session_state.segments_fingerprint = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def compute_fingerprint(df: pl.DataFrame) -> str:
    """Compute a stable fingerprint of the DataFrame for change detection.
    Uses a hash of the underlying file for deterministic fingerprinting."""
    # For Parquet files, we can't easily get the file path from a DataFrame
    # Fall back to hashing a stable sample
    # Sort by all columns for deterministic ordering
    df_sorted = df.sort(df.columns)
    n_rows = df_sorted.height
    schema_str = str(df_sorted.schema)
    sample = df_sorted.head(min(1000, n_rows)).to_numpy().tobytes()
    content = f"{n_rows}:{schema_str}:{hashlib.md5(sample).hexdigest()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]

def compute_file_fingerprint(uploaded_file) -> str:
    """Compute fingerprint of uploaded file bytes."""
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.sha256(content).hexdigest()[:16]

def compute_local_file_fingerprint(path: str) -> str:
    """Compute fingerprint of local Parquet file."""
    import hashlib
    with open(path, 'rb') as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()[:16]

# ============================================================
# FILE UPLOAD
# ============================================================
st.sidebar.header("📁 Data Upload")
st.sidebar.markdown("Upload **cleaned transaction data** only. The app will generate segments on demand.")

uploaded_transactions = st.sidebar.file_uploader(
    "Transactions Parquet (customer_transactions_clean.parquet)",
    type=["parquet"],
    help="Cleaned transaction data - the app will generate segments automatically"
)

# ============================================================
# DATA LOADERS (cached)
# ============================================================
@st.cache_data
def load_transactions_from_file(uploaded_file) -> pl.DataFrame:
    """Load transactions from uploaded file or local fallback."""
    if uploaded_file is not None:
        return pl.read_parquet(io.BytesIO(uploaded_file.read()))
    # Fallback to local file
    try:
        return pl.read_parquet("customer_transactions_clean.parquet")
    except FileNotFoundError:
        st.warning("No transactions file found. Please upload a parquet file.")
        return None

# ============================================================
# SEGMENTATION PIPELINE (cached) - returns (segments_df, metadata_dict)
# ============================================================
@st.cache_data(show_spinner=False)
def run_segmentation_pipeline(transactions_df: pl.DataFrame, tx_fingerprint: str):
    """Run the full segmentation pipeline on transaction data.
    Returns (segments_df, metadata_dict)."""
    pipeline = AdvancedCustomerSegmentation(max_clusters=8, n_components=6)
    segments = pipeline.fit_predict(transactions_df)
    metadata = pipeline.get_pca_metadata(tx_fingerprint)
    return segments, metadata

def get_or_generate_segments(tx_df: pl.DataFrame, tx_fingerprint: str, force_refresh: bool = False):
    """Load existing segments or generate new ones from transactions.
    Segments are matched to source data via fingerprint.
    force_refresh=True bypasses cached artifact."""
    
    # Check if we have a matching local segment artifact
    if not force_refresh:
        try:
            segments = pl.read_parquet("customer_segments_advanced.parquet")
            # Check if fingerprint matches
            try:
                with open("pca_metadata.json", 'r') as f:
                    metadata = json.load(f)
                if metadata.get('source_fingerprint') == tx_fingerprint:
                    st.sidebar.success("✅ Loaded matching segments from local file")
                    return segments, metadata
                else:
                    st.sidebar.warning("⚠️ Local segments don't match current data. Regenerating...")
            except (FileNotFoundError, json.JSONDecodeError):
                st.sidebar.warning("⚠️ No metadata for local segments. Regenerating...")
        except FileNotFoundError:
            pass
    
    # If no matching local file, or force_refresh, run pipeline on provided transactions
    if tx_df is not None:
        status_placeholder = st.sidebar.empty()
        try:
            with st.sidebar.status("Running segmentation pipeline...", expanded=True) as status:
                status.write("Step 1/5: Validating and cleaning data...")
                status.write("Step 2/5: Engineering 50+ behavioral metrics...")
                status.write("Step 3/5: Winsorization, RobustScaling & PCA...")
                status.write("Step 4/5: Optimizing cluster count (Davies-Bouldin)...")
                status.write("Step 5/5: Fitting final KMeans model...")
                
                segments, metadata = run_segmentation_pipeline(tx_df, tx_fingerprint)
                
                # Save to local file for future runs
                segments.write_parquet("customer_segments_advanced.parquet")
                with open("pca_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                status.update(label="✅ Segments generated and saved!", state="complete")
            
            return segments, metadata
        except Exception as e:
            st.sidebar.error(f"❌ Segmentation failed: {str(e)}")
            st.error(f"Segmentation error: {e}")
            st.stop()
    
    st.error("No transaction data available. Please upload a parquet file.")
    st.stop()

# ============================================================
# RETENTION MATRIX (cached) - Fixed: Proper cohort maturity handling
# ============================================================
@st.cache_data
def generate_retention_matrix_polars(transactions_lf: pl.LazyFrame) -> pl.DataFrame:
    """Generates a Monthly Repeat-Purchase Retention matrix entirely in Polars.
    Properly handles cohort maturity: immature periods = null, mature zero = 0%."""
    df = transactions_lf.with_columns([
        pl.col('transaction_date').dt.truncate('1mo').alias('OrderMonth'),
    ]).with_columns([
        pl.col('transaction_date').dt.truncate('1mo').alias('tx_month'),
    ])
    
    cohort_months = df.group_by('customer_id').agg(
        pl.col('tx_month').min().alias('CohortMonth')
    )
    
    df = df.join(cohort_months, on='customer_id', how='left')
    
    df = df.with_columns([
        ((pl.col('OrderMonth').dt.year() - pl.col('CohortMonth').dt.year()) * 12 + 
         (pl.col('OrderMonth').dt.month() - pl.col('CohortMonth').dt.month())).alias('PeriodNumber')
    ])
    
    cohort_data = df.group_by(['CohortMonth', 'PeriodNumber']).agg(
        pl.col('customer_id').n_unique().alias('customer_count')
    ).collect()
    
    cohort_pivot = cohort_data.pivot(
        values='customer_count',
        index='CohortMonth',
        on='PeriodNumber',
        aggregate_function='first'
    ).sort('CohortMonth')
    
    # DO NOT fill_null globally - preserves null for unobserved periods
    period_cols = [c for c in cohort_pivot.columns if c != 'CohortMonth' and isinstance(c, (int, str))]
    period_cols_int = []
    for c in period_cols:
        try:
            period_cols_int.append(int(c))
        except (ValueError, TypeError):
            pass
    
    if period_cols_int:
        period_cols_int.sort()
        
        # Calculate dataset's final observed month for maturity
        max_order_month = df.select(pl.col('OrderMonth').max()).collect().item()
        
        if 0 in period_cols_int:
            cohort_sizes = cohort_pivot.select(pl.col("0")).to_series()
            for pc in period_cols_int:
                pc_str = str(pc)
                if pc != 0:
                    # Determine if period is mature for each cohort
                    # A period is mature if CohortMonth + pc <= max_order_month
                    # For each cohort row, check maturity
                    cohort_pivot = cohort_pivot.with_columns(
                        pl.when(
                            (pl.col("0") > 0) & 
                            ((pl.col('CohortMonth').dt.year() * 12 + pl.col('CohortMonth').dt.month() + pc) <= 
                             (max_order_month.year() * 12 + max_order_month.month()))
                        )
                        .then(
                            pl.when(pl.col(pc_str).is_not_null())
                            .then(pl.col(pc_str) / pl.col("0"))
                            .otherwise(pl.lit(0.0))
                        )
                        .otherwise(None)
                        .alias(pc_str)
                    )
            # Month 0 is always 100% where cohort exists
            cohort_pivot = cohort_pivot.with_columns(
                pl.when(pl.col("0") > 0).then(pl.lit(1.0)).otherwise(None).alias("0")
            )
    
    cohort_pivot = cohort_pivot.with_columns(
        pl.col('CohortMonth').dt.strftime('%Y-%m').alias('Cohort')
    ).drop('CohortMonth')
    
    # Ensure period columns are sorted numerically
    new_cols = ['Cohort'] + [str(pc) for pc in period_cols_int]
    return cohort_pivot.select(new_cols)

# ============================================================
# MAIN DATA LOADING LOGIC
# ============================================================

# Load transactions (handle upload changes correctly)
if uploaded_transactions is not None:
    # New upload - compute fingerprint to detect actual changes
    new_fp = compute_file_fingerprint(uploaded_transactions)
    
    if st.session_state.tx_fingerprint != new_fp:
        # Data actually changed - load new data and reset segments
        tx_df = pl.read_parquet(io.BytesIO(uploaded_transactions.read()))
        st.session_state.tx_df = tx_df
        st.session_state.tx_fingerprint = new_fp
        st.session_state.segments_generated = False
        st.session_state.segments_df = None
        st.session_state.segments_fingerprint = None
        st.sidebar.info("🔄 New transaction data detected. Please regenerate segments.")
    else:
        # Same data - use cached
        tx_df = st.session_state.tx_df
else:
    # No upload - try local file or session state
    if st.session_state.tx_df is not None:
        tx_df = st.session_state.tx_df
    else:
        try:
            tx_df = pl.read_parquet("customer_transactions_clean.parquet")
            st.session_state.tx_df = tx_df
            if st.session_state.tx_fingerprint is None:
                st.session_state.tx_fingerprint = compute_local_file_fingerprint("customer_transactions_clean.parquet")
        except FileNotFoundError:
            st.error("No transaction data available. Please upload a parquet file.")
            st.stop()

tx_lazy = tx_df.lazy()

# ============================================================
# SEGMENTATION TRIGGER
# ============================================================
if not st.session_state.segments_generated or st.session_state.segments_df is None:
    # Show generate button
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Generate / Refresh Segments", type="primary", use_container_width=True):
            with st.spinner("Running segmentation..."):
                segments_df, metadata = get_or_generate_segments(tx_df, st.session_state.tx_fingerprint, force_refresh=True)
                st.session_state.segments_df = segments_df
                st.session_state.segments_fingerprint = st.session_state.tx_fingerprint
                st.session_state.segments_generated = True
            st.rerun()
    with col2:
        if st.button("⚡ Force", help="Force regenerate even if segments exist"):
            with st.spinner("Force regenerating..."):
                segments_df, metadata = get_or_generate_segments(tx_df, st.session_state.tx_fingerprint, force_refresh=True)
                st.session_state.segments_df = segments_df
                st.session_state.segments_fingerprint = st.session_state.tx_fingerprint
                st.session_state.segments_generated = True
            st.rerun()
    st.info("👈 Click **'Generate / Refresh Segments'** in the sidebar to run the segmentation pipeline.")
    st.stop()
else:
    segments_df = st.session_state.segments_df

# Verify segments match current data
if st.session_state.segments_fingerprint != st.session_state.tx_fingerprint:
    st.sidebar.error("⚠️ Segments don't match current transaction data! Please refresh.")
    st.session_state.segments_generated = False
    st.rerun()

# ============================================================
# POST-PROCESSING: Add Primary Category
# ============================================================
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
    return df.with_columns(
        pl.col('Cluster').cast(pl.Utf8).map_elements(
            lambda x: f"Cluster {x}", return_dtype=pl.Utf8
        ).alias('Segment')
    )

def filter_empty_warning(filtered_df: pl.DataFrame) -> bool:
    if filtered_df.is_empty():
        st.info("Select at least one cluster to explore customer behavior.")
        st.stop()
        return True
    return False

# ============================================================
# UI: SIDEBAR FILTERS
# ============================================================
st.sidebar.header("🎯 Global Filters")
selected_clusters = st.sidebar.multiselect(
    "Select Clusters", 
    clusters, 
    default=clusters,
    format_func=lambda x: f"Cluster {x}"
)
filtered_df = segments_df.filter(pl.col('Cluster').is_in(selected_clusters))

filter_empty_warning(filtered_df)

# Add Segment column for plotting
filtered_df = add_segment_column(filtered_df)
segments_df = add_segment_column(segments_df)

# ============================================================
# SUMMARY CARDS (use filtered_df)
# ============================================================
total_customers = segments_df.height
selected_count = filtered_df.height
selected_share = selected_count / total_customers * 100
selected_spend = filtered_df.select(pl.col('Total_Lifetime_Spend').sum()).item()
median_aov = filtered_df.select(pl.col('Average_Order_Value_AOV').median()).item()
median_recency = filtered_df.select(pl.col('Recency_Days').median()).item()

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Risk–Value Portfolio", 
    "Segment Profiles", 
    "Distributions", 
    "Model Exploration",
    "Retention"
])

# ------------------------------------------------------------------
# TAB 1: RISK–VALUE PORTFOLIO (Fixed: aligned spend threshold)
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
    
    # Use full dataset for threshold calculation (not filtered)
    full_plot_df = segments_df.select([
        'customer_id', 'Cluster', 'Segment', 'Total_Lifetime_Spend', 
        'Churn_Risk_Index_CRI', 'Recency_Days', 
        'Average_InterPurchase_Time_AIPT', 'Average_Order_Value_AOV'
    ]).to_pandas()
    
    has_history = full_plot_df['Average_InterPurchase_Time_AIPT'] > 0
    no_history = full_plot_df[~has_history]
    full_plot_df = full_plot_df[has_history]
    
    # Apply cluster filter for plotting
    plot_df = full_plot_df[full_plot_df['Cluster'].isin(selected_clusters)]
    
    col1, col2 = st.columns([3, 1])
    with col2:
        log_scale = st.checkbox("Log scale (Y-axis)", value=True)
        show_no_history = st.checkbox("Show 'Insufficient history'", value=True)
        spend_pct = st.slider("High-value threshold (percentile)", 50, 99, 80, help="Horizontal line separating high-value customers")
    
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
    
    # Vertical reference lines
    fig_scatter.add_vline(x=1.0, line_dash="dot", line_color="gray", 
                          annotation_text="On Cadence", annotation_position="top left")
    fig_scatter.add_vline(x=2.0, line_dash="dot", line_color="orange", 
                          annotation_text="2× Cadence", annotation_position="top left")
    fig_scatter.add_vline(x=3.0, line_dash="dash", line_color="red", 
                          annotation_text="Critical Delay", annotation_position="top right")
    
    # Horizontal spend threshold (percentile-based) - calculated on FULL dataset
    spend_threshold = full_plot_df['Total_Lifetime_Spend'].quantile(spend_pct / 100)
    fig_scatter.add_hline(y=spend_threshold, line_dash="dot", line_color="green",
                          annotation_text=f"High Value ({spend_pct}th pctl: ${spend_threshold:,.0f})", 
                          annotation_position="bottom right")
    
    # Quadrant annotations - properly aligned with risk zones
    fig_scatter.add_annotation(x=0.5, y=spend_threshold * 1.2, 
                              text="✅ High Value + On Cadence", showarrow=False,
                              font=dict(size=10, color="green"), bgcolor="rgba(255,255,255,0.8)")
    fig_scatter.add_annotation(x=1.5, y=spend_threshold * 1.2, 
                              text="⚠️ High Value + Early Delay", showarrow=False,
                              font=dict(size=10, color="orange"), bgcolor="rgba(255,255,255,0.8)")
    fig_scatter.add_annotation(x=2.5, y=spend_threshold * 1.2, 
                              text="🔴 High Value + Overdue", showarrow=False,
                              font=dict(size=10, color="red"), bgcolor="rgba(255,255,255,0.8)")
    fig_scatter.add_annotation(x=0.5, y=spend_threshold * 0.3,
                              text="Low Value + On Cadence", showarrow=False,
                              font=dict(size=10, color="gray"), bgcolor="rgba(255,255,255,0.8)")
    fig_scatter.add_annotation(x=2.5, y=spend_threshold * 0.3,
                              text="Low Value + Overdue", showarrow=False,
                              font=dict(size=10, color="orange"), bgcolor="rgba(255,255,255,0.8)")
    
    if show_no_history and not no_history.empty:
        fig_scatter.add_scatter(
            x=no_history['Churn_Risk_Index_CRI'],
            y=no_history['Total_Lifetime_Spend'],
            mode='markers',
            marker=dict(symbol='x', size=8, color='lightgray'),
            name='Insufficient history',
            showlegend=True,
            hovertext=no_history['customer_id'].astype(str) + ': $' + no_history['Total_Lifetime_Spend'].astype(str)
        )
    
    fig_scatter.update_layout(
        xaxis_title="Purchase-Delay Index (Recency ÷ Avg Inter-Purchase Days)",
        yaxis_title="Spend in Observed History ($)",
        height=600,
        legend_title="Segment"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.subheader("Top Targets: High Value + High Delay")
    # Use SAME threshold as chart
    high_delay = full_plot_df[
        (full_plot_df['Churn_Risk_Index_CRI'] >= 2.0) & 
        (full_plot_df['Total_Lifetime_Spend'] >= spend_threshold)
    ].nlargest(20, 'Total_Lifetime_Spend')
    
    if not high_delay.empty:
        st.dataframe(high_delay[['customer_id', 'Segment', 'Total_Lifetime_Spend', 
                                  'Churn_Risk_Index_CRI', 'Recency_Days', 'Average_Order_Value_AOV']])
    else:
        st.info("No customers meeting both high-value and high-delay criteria.")

# ------------------------------------------------------------------
# TAB 2: SEGMENT PROFILES (Fixed: Correct hover z-scores)
# ------------------------------------------------------------------
with tab2:
    st.header("Segment Profiles")
    st.markdown("""
    **How does each segment differ from the overall customer base?**
    
    Colors show standardized differences (segment mean − overall mean) ÷ overall std.
    Red = above average, Blue = below average. White = near overall average.
    """)
    
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
        # Overall benchmarks from FULL dataset
        overall_means = segments_df.select([pl.col(m).mean().alias(m) for m in available_metrics]).to_pandas().iloc[0]
        overall_stds = segments_df.select([pl.col(m).std().alias(m) for m in available_metrics]).to_pandas().iloc[0]
        
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
        
        # Prepare customdata for hover - ONE value per cell: [raw_mean, overall_mean, z_score]
        hover_data = []
        for _, row in z_data.iterrows():
            cell_data = []
            for i, m in enumerate(available_metrics):
                raw = row[m]
                overall = overall_means[m]
                std_val = overall_stds[m]
                z_score = (raw - overall) / std_val if std_val > 0 else 0
                cell_data.append([raw, overall, z_score])
            hover_data.append(cell_data)
        
        # hover_data is now list of lists: each cell has [raw, overall, z]
        # Convert to numpy array with shape (n_segments, n_metrics, 3)
        hover_array = np.array(hover_data)
        
        z_values = z_data[available_metrics].values
        
        fig_heatmap = px.imshow(
            z_values,
            x=[metric_labels[m] for m in available_metrics],
            y=z_data['Segment'].tolist(),
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            text_auto=".2f",
            aspect="auto",
            title="Standardized Segment Differences from Customer Base",
            # Set symmetric color range for consistency across selections
            color_continuous_scale='RdBu_r',
        )
        # Add custom hover with raw values
        fig_heatmap.update_traces(
            customdata=hover_array,
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Metric: %{x}<br>" +
                "Segment mean: %{customdata[0]:.2f}<br>" +
                "Overall mean: %{customdata[1]:.2f}<br>" +
                "Z-score: %{customdata[2]:.2f}" +
                "<extra></extra>"
            )
        )
        fig_heatmap.update_layout(
            xaxis_title="",
            yaxis_title="",
            height=400 + len(z_data) * 30,
            coloraxis_colorbar=dict(title="Std. Diff. from Overall"),
            # Ensure symmetric color range
            coloraxis_cmin=-3,
            coloraxis_cmax=3,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.caption("""
        **How to read:** Values are (segment mean − overall mean) / overall std. 
        A value of +1.0 means the segment averages one standard deviation above the customer base.
        Red = above average, Blue = below average. Hover for raw values and benchmarks.
        """)
        
        with st.expander("Radar View (min-max normalized within selection)"):
            st.markdown("*Shape depends on filtered selection; use heatmap above for stable comparison.*")
            
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
# TAB 3: DISTRIBUTIONS
# ------------------------------------------------------------------
with tab3:
    st.header("Distributions Within Segments")
    st.markdown("Cluster averages hide variation. Select a metric to see the full distribution per segment.")
    
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
# TAB 4: MODEL EXPLORATION (Fixed: Actual PCA metadata)
# ------------------------------------------------------------------
with tab4:
    st.header("Model Exploration: PCA Projection")
    st.markdown("""
    Diagnostic view of the clustering in principal component space.
    
    - **Components**: First 3 principal components from robust PCA (6 components)
    - **Explained variance**: Actual values from pipeline metadata
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
                scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
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
        else:
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
            # Try to load actual PCA metadata
            try:
                with open("pca_metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                evr = metadata.get('explained_variance_ratio', [])
                evc = metadata.get('explained_variance_cumulative', [])
                
                if evr:
                    # Plot explained variance
                    fig_ev = go.Figure()
                    fig_ev.add_trace(go.Bar(
                        x=[f'PC{i+1}' for i in range(len(evr))],
                        y=[v * 100 for v in evr],
                        name='Individual',
                        marker_color='steelblue',
                        text=[f'{v*100:.1f}%' for v in evr],
                        textposition='outside',
                    ))
                    fig_ev.add_trace(go.Scatter(
                        x=[f'PC{i+1}' for i in range(len(evc))],
                        y=[v * 100 for v in evc],
                        name='Cumulative',
                        mode='lines+markers+text',
                        line=dict(color='red', width=3),
                        text=[f'{v*100:.1f}%' for v in evc],
                        textposition='top center',
                        yaxis='y2'
                    ))
                    fig_ev.update_layout(
                        title="PCA Explained Variance",
                        xaxis_title="Principal Component",
                        yaxis=dict(title="Individual Variance (%)", side='left'),
                        yaxis2=dict(title="Cumulative Variance (%)", side='right', overlaying='y', range=[0, 100]),
                        height=400,
                        legend=dict(x=0.02, y=0.98)
                    )
                    st.plotly_chart(fig_ev, use_container_width=True)
                    
                    st.caption(f"""
                    **Total explained variance (6 components): {evc[-1]*100:.1f}%**
                    - PC1: {evr[0]*100:.1f}% (Spend & Volume)
                    - PC2: {evr[1]*100:.1f}% (Cadence & Frequency)
                    - PC3: {evr[2]*100:.1f}% (Category Depth)
                    - PC4: {evr[3]*100:.1f}% (Basket Variance)
                    - PC5: {evr[4]*100:.1f}% (Discount Affinity)
                    - PC6: {evr[5]*100:.1f}% (Price Variability)
                    """)
                else:
                    st.info("PCA metadata not available. Run segmentation to generate.")
            except (FileNotFoundError, json.JSONDecodeError):
                st.info("PCA metadata not found. Run segmentation to generate pca_metadata.json.")
    else:
        st.warning("PCA components (PC1, PC2, PC3) not found in segment data. Re-run segmentation to generate them.")

# ------------------------------------------------------------------
# TAB 5: RETENTION (Fixed: Proper cohort maturity)
# ------------------------------------------------------------------
with tab5:
    st.header("Monthly Repeat-Purchase Retention")
    st.markdown("""
    Tracks the fraction of each first-observed-purchase cohort that makes at least one purchase in each subsequent month.
    
    - **Cohort**: Month of customer's first observed purchase
    - **Period**: Months since first observed purchase (0 = cohort month)
    - **Value**: % of cohort that purchased in that period
    - **Missing (white)**: Period not yet mature enough to observe
    - **0%**: Mature period with no returning customers
    """)
    
    if tx_lazy is None:
        st.warning("No transactions data loaded. Retention unavailable.")
    else:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Generate Retention Matrix", type="primary", use_container_width=True):
                st.session_state.show_retention = True
            show_retention = st.session_state.get('show_retention', False)
        
        if st.session_state.get('show_retention', False):
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
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    period_cols = [c for c in retention_df.columns if c != 'Cohort']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if '1' in period_cols:
                            # Only include non-null (mature) values in average
                            m1 = retention_df.select(pl.col('1').drop_nulls().mean()).item()
                            st.metric("Avg Month 1 Retention (mature)", f"{m1:.1%}")
                    with col2:
                        if '3' in period_cols:
                            m3 = retention_df.select(pl.col('3').drop_nulls().mean()).item()
                            st.metric("Avg Month 3 Retention (mature)", f"{m3:.1%}")
                    with col3:
                        st.metric("Cohorts", retention_df.height)
                else:
                    st.warning("Insufficient transaction history for cohort analysis.")

# ------------------------------------------------------------------
# SIDEBAR: Segment Summary (Fixed: iterate selected_clusters, use full segments_df)
# ------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Segment Summary")

# Use selected_clusters for iteration, compute from segments_df (full) for stable metrics
for c in selected_clusters:
    subset = segments_df.filter(pl.col('Cluster') == c)
    if subset.is_empty():
        continue
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

if not selected_clusters:
    st.sidebar.info("No clusters selected")