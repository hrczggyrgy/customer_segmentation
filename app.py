import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Advanced Customer Analytics", layout="wide")
st.title("Customer Intelligence Platform")

# 2. Data Loaders (Cached for performance)
@st.cache_data
def load_segments():
    return pl.read_parquet("customer_segments_advanced.parquet").to_pandas()

@st.cache_data
def load_transactions():
    return pl.read_parquet("customer_transactions_clean.parquet").to_pandas()

@st.cache_data
def generate_retention_matrix(transactions_df):
    """Generates a Triangular Cohort Retention Heatmap matrix."""
    df = transactions_df.copy()
    
    # Create Year-Month columns for the transaction and the customer's first purchase
    df['OrderMonth'] = df['transaction_date'].dt.to_period('M')
    df['CohortMonth'] = df.groupby('customer_id')['transaction_date'].transform('min').dt.to_period('M')
    
    # Group by Cohort and Order Month, count unique customers
    cohort_data = df.groupby(['CohortMonth', 'OrderMonth'])['customer_id'].nunique().reset_index()
    
    # Calculate the exact period distance (Month 0, Month 1, etc.)
    cohort_data['PeriodNumber'] = (cohort_data.OrderMonth - cohort_data.CohortMonth).apply(lambda x: x.n)
    
    # Pivot the data into a matrix
    cohort_pivot = cohort_data.pivot_table(index='CohortMonth', columns='PeriodNumber', values='customer_id')
    
    # Divide by the initial cohort size (Month 0) to get percentages
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    
    # Convert index/columns to strings for Plotly
    retention_matrix.index = retention_matrix.index.astype(str)
    retention_matrix.columns = retention_matrix.columns.astype(str)
    
    return retention_matrix

# Load data
df = load_segments()
tx_df = load_transactions()

# Add Primary_Category_Name if missing (derive from transactions)
if 'Primary_Category_Name' not in df.columns:
    primary_cat = tx_df.groupby('customer_id')['category'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
    ).reset_index()
    primary_cat.columns = ['customer_id', 'Primary_Category_Name']
    df = df.merge(primary_cat, on='customer_id', how='left')
    df['Primary_Category_Name'] = df['Primary_Category_Name'].fillna('Unknown')

# Sidebar Filters
st.sidebar.header("Global Filters")
clusters = sorted(df['Cluster'].unique())
selected_clusters = st.sidebar.multiselect("Select Clusters", clusters, default=clusters)
filtered_df = df[df['Cluster'].isin(selected_clusters)]

# 3. Tab Layout
tab1, tab2, tab3 = st.tabs(["Cluster Topography", "Behavioral Heatmaps", "Cohort Retention"])

with tab1:
    st.header("3D Cluster Projection")
    st.markdown("Visualizing the mathematical boundaries of your customer segments in PCA space.")
    
    # Require PCA components to be in the dataframe (e.g., PC1, PC2, PC3)
    pca_cols = [c for c in df.columns if c.startswith('PC')]
    if len(pca_cols) >= 3:
        fig_3d = px.scatter_3d(
            filtered_df, x='PC1', y='PC2', z='PC3',
            color='Cluster',
            hover_data=['customer_id', 'Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Primary_Category_Name'],
            size='Total_Lifetime_Spend',
            size_max=20,
            opacity=0.7,
            title="3D Behavioral PCA Space",
            color_continuous_scale='Set2' if filtered_df['Cluster'].nunique() <= 10 else 'Viridis'
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
        
        # Show explained variance if available
        st.caption(f"PCA Components available: {', '.join(pca_cols)}")
    else:
        st.warning("PCA components (PC1, PC2, PC3) not found in segment data. Re-run segmentation.py to generate them.")
        st.info("The updated segmentation pipeline now exports PCA components automatically.")

with tab2:
    st.header("Cluster Feature Profiling")
    st.markdown("Comparing the average behavioral traits that define each segment.")
    
    # Select key metrics for profiling
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
    
    # Filter to only available columns
    available_metrics = [m for m in metrics_to_plot if m in filtered_df.columns]
    
    if available_metrics:
        # Calculate the mean of key metrics per cluster
        cluster_means = filtered_df.groupby('Cluster')[available_metrics].mean().reset_index()
        
        # Melt for Plotly Barchart
        melted_means = cluster_means.melt(id_vars='Cluster', var_name='Metric', value_name='Average Value')
        
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
        
        # Radar chart for cluster comparison
        st.subheader("Segment Personas (Normalized Radar)")
        
        # Normalize metrics for radar chart (0-1 scale per metric)
        radar_df = cluster_means.copy()
        for col in available_metrics:
            max_val = radar_df[col].max()
            min_val = radar_df[col].min()
            if max_val > min_val:
                radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)
            else:
                radar_df[col] = 0.5
        
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
    
    # Generate retention matrix
    retention_matrix = generate_retention_matrix(tx_df)
    
    if not retention_matrix.empty:
        # Plotly Heatmap
        fig_heatmap = px.imshow(
            retention_matrix.values,
            x=retention_matrix.columns,
            y=retention_matrix.index,
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
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            # Average Month 1 retention
            if '1' in retention_matrix.columns:
                m1_retention = retention_matrix['1'].mean()
                st.metric("Avg Month 1 Retention", f"{m1_retention:.1%}")
        with col2:
            # Average Month 3 retention
            if '3' in retention_matrix.columns:
                m3_retention = retention_matrix['3'].mean()
                st.metric("Avg Month 3 Retention", f"{m3_retention:.1%}")
        with col3:
            # Number of cohorts
            st.metric("Active Cohorts", len(retention_matrix))
    else:
        st.warning("Insufficient transaction history for cohort analysis.")

# Additional: Quick Segment Summary in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Segment Summary")
for c in clusters:
    subset = filtered_df[filtered_df['Cluster'] == c]
    st.sidebar.markdown(f"""
    **Cluster {c}** ({len(subset)} customers)
    - Avg Spend: ${subset['Total_Lifetime_Spend'].mean():,.0f}
    - Avg AOV: ${subset['Average_Order_Value_AOV'].mean():.2f}
    - Churn Risk: {subset['Churn_Risk_Index_CRI'].mean():.2f}
    - Top Category: {subset['Primary_Category_Name'].mode().iloc[0] if not subset.empty else 'N/A'}
    """)