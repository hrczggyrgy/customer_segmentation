import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Advanced Customer Analytics", layout="wide")
st.title("Customer Intelligence Platform")

# 2. Data Loaders (Cached for performance) - Keep in Polars!
@st.cache_data
def load_segments():
    return pl.read_parquet("customer_segments_advanced.parquet")

@st.cache_data
def load_transactions():
    return pl.read_parquet("customer_transactions_clean.parquet")

@st.cache_data
def generate_retention_matrix_polars(transactions_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Generates a Triangular Cohort Retention Heatmap matrix entirely in Polars.
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
    
    # Fill nulls with 0
    cohort_pivot = cohort_pivot.fill_null(0)
    
    # Convert to retention rates (divide by Month 0 size)
    period_cols = [c for c in cohort_pivot.columns if c != 'CohortMonth' and isinstance(c, int)]
    if 0 in period_cols:
        cohort_sizes = cohort_pivot.select(pl.col(0)).to_series()
        for pc in period_cols:
            if pc != 0:
                cohort_pivot = cohort_pivot.with_columns(
                    (pl.col(pc) / cohort_sizes).alias(pc)
                )
        # Month 0 is always 100%
        cohort_pivot = cohort_pivot.with_columns(pl.lit(1.0).alias(0))
    
    # Convert CohortMonth to string for display
    cohort_pivot = cohort_pivot.with_columns(
        pl.col('CohortMonth').dt.strftime('%Y-%m').alias('Cohort')
    ).drop('CohortMonth')
    
    return cohort_pivot

# Load data (LazyFrame for transactions, DataFrame for segments)
segments_df = load_segments()
tx_lazy = load_transactions().lazy()

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

# Sidebar Filters
st.sidebar.header("Global Filters")
clusters = sorted(segments_df['Cluster'].unique().to_list())
selected_clusters = st.sidebar.multiselect("Select Clusters", clusters, default=clusters)
filtered_df = segments_df.filter(pl.col('Cluster').is_in(selected_clusters))

# 3. Tab Layout
tab1, tab2, tab3 = st.tabs(["Cluster Topography", "Behavioral Heatmaps", "Cohort Retention"])

with tab1:
    st.header("3D Cluster Projection")
    st.markdown("Visualizing the mathematical boundaries of your customer segments in PCA space.")
    
    pca_cols = [c for c in segments_df.columns if c.startswith('PC')]
    if len(pca_cols) >= 3:
        # Convert filtered subset to Pandas for Plotly
        plot_df = filtered_df.select(['customer_id', 'Cluster', 'Total_Lifetime_Spend', 
                                       'Average_Order_Value_AOV', 'Primary_Category_Name',
                                       'PC1', 'PC2', 'PC3']).to_pandas()
        
        fig_3d = px.scatter_3d(
            plot_df, x='PC1', y='PC2', z='PC3',
            color='Cluster',
            hover_data=['customer_id', 'Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Primary_Category_Name'],
            size='Total_Lifetime_Spend',
            size_max=20,
            opacity=0.7,
            title="3D Behavioral PCA Space",
            color_continuous_scale='Set2' if filtered_df['Cluster'].n_unique() <= 10 else 'Viridis'
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='PC1: Spend & Volume',
                yaxis_title='PC2: Cadence & Frequency',
                zaxis_title='PC3: Category Depth'
            ),
            height=700
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption(f"PCA Components available: {', '.join(pca_cols)}")
    else:
        st.warning("PCA components (PC1, PC2, PC3) not found in segment data. Re-run segmentation.py to generate them.")

with tab2:
    st.header("Cluster Feature Profiling")
    st.markdown("Comparing the average behavioral traits that define each segment.")
    
    metrics_to_plot = [
        'Average_Order_Value_AOV', 
        'Churn_Risk_Index_CRI', 
        'Category_Width', 
        'Total_Lifetime_Transactions',
        'Department_Penetration_Pct',
        'Active_Days_Ratio',
        'Average_InterPurchase_Time_AIPT',
        'Inferred_Discount_Affinity'
    ]
    
    available_metrics = [m for m in metrics_to_plot if m in filtered_df.columns]
    
    if available_metrics:
        # Compute cluster means in Polars
        cluster_means = filtered_df.group_by('Cluster').agg([
            pl.col(m).mean().alias(m) for m in available_metrics
        ]).sort('Cluster')
        
        # Convert to Pandas for Plotly
        cluster_means_pd = cluster_means.to_pandas()
        
        # Melt for grouped bar chart
        melted_means = cluster_means_pd.melt(id_vars='Cluster', var_name='Metric', value_name='Average Value')
        
        fig_bar = px.bar(
            melted_means, 
            x='Metric', 
            y='Average Value', 
            color='Cluster', 
            barmode='group',
            title="Average Behavioral Traits by Segment",
            text='Average Value'
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_layout(height=600, xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Radar chart - Normalize in Polars first
        st.subheader("Segment Personas (Normalized Radar)")
        
        # Min-max normalize each metric across clusters
        radar_exprs = []
        for col in available_metrics:
            max_val = cluster_means.select(pl.col(col).max()).item()
            min_val = cluster_means.select(pl.col(col).min()).item()
            if max_val > min_val:
                radar_exprs.append(((pl.col(col) - min_val) / (max_val - min_val)).alias(col))
            else:
                radar_exprs.append(pl.lit(0.5).alias(col))
        
        radar_df = cluster_means.with_columns(radar_exprs).to_pandas()
        
        fig_radar = go.Figure()
        for _, row in radar_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=row[available_metrics].values,
                theta=available_metrics,
                fill='toself',
                name=f"Cluster {int(row['Cluster'])}"
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500,
            title="Normalized Cluster Profiles (0-1 scale per metric)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("Required metrics not found in segment data.")

with tab3:
    st.header("Lifecycle Decay (Cohort Retention)")
    st.markdown("Tracks the survival rate of customer acquisition cohorts over time.")
    
    # Generate retention matrix in Polars
    retention_df = generate_retention_matrix_polars(tx_lazy)
    
    if retention_df.height > 0 and retention_df.width > 1:
        # Convert to Pandas for Plotly heatmap
        retention_pd = retention_df.to_pandas()
        retention_pd = retention_pd.set_index('Cohort')
        
        fig_heatmap = px.imshow(
            retention_pd.values,
            x=retention_pd.columns.astype(str),
            y=retention_pd.index,
            text_auto=".0%",
            color_continuous_scale="YlGnBu",
            title="Cohort Retention Heatmap",
            aspect="auto"
        )
        fig_heatmap.update_layout(
            xaxis_title="Months Since First Purchase",
            yaxis_title="Acquisition Cohort",
            height=600
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Summary metrics computed in Polars
        period_cols = [c for c in retention_df.columns if c != 'Cohort']
        col1, col2, col3 = st.columns(3)
        with col1:
            if '1' in period_cols:
                m1_retention = retention_df.select(pl.col('1').mean()).item()
                st.metric("Avg Month 1 Retention", f"{m1_retention:.1%}")
        with col2:
            if '3' in period_cols:
                m3_retention = retention_df.select(pl.col('3').mean()).item()
                st.metric("Avg Month 3 Retention", f"{m3_retention:.1%}")
        with col3:
            st.metric("Active Cohorts", retention_df.height)
    else:
        st.warning("Insufficient transaction history for cohort analysis.")

# Sidebar Segment Summary - computed in Polars
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
    - Churn Risk: {churn_risk:.2f}
    - Top Category: {top_cat}
    """)