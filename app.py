# app.py
import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Retail Segmentation CRM", layout="wide")
st.title("Customer Segmentation & Action Dashboard")

# 2. Data Loader (Cached for performance)
@st.cache_data
def load_data():
    # Polars reads the parquet file instantly
    return pl.read_parquet("customer_segments_advanced.parquet").to_pandas()

@st.cache_data
def load_transactions():
    return pl.read_parquet("customer_transactions_clean.parquet").to_pandas()

df = load_data()
clusters = sorted(df['Cluster'].unique())

# Add Primary_Category_Name if missing (derive from transactions)
if 'Primary_Category_Name' not in df.columns:
    tx = load_transactions()
    primary_cat = tx.groupby('customer_id')['category'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown').reset_index()
    primary_cat.columns = ['customer_id', 'Primary_Category_Name']
    df = df.merge(primary_cat, on='customer_id', how='left')
    df['Primary_Category_Name'] = df['Primary_Category_Name'].fillna('Unknown')

# 3. Sidebar Filters
st.sidebar.header("Global Filters")
selected_clusters = st.sidebar.multiselect("Select Clusters", clusters, default=clusters)
filtered_df = df[df['Cluster'].isin(selected_clusters)]

# 4. Tab Layout
tab1, tab2, tab3 = st.tabs(["Macro Overview", "Risk vs. Value Action Center", "CRM Export"])

with tab1:
    st.header("Segment Population & Value")
    col1, col2 = st.columns(2)
    
    with col1:
        # Population Chart
        pop_counts = filtered_df['Cluster'].value_counts().reset_index()
        fig_pop = px.pie(pop_counts, values='count', names='Cluster', title='Customer Count by Segment')
        st.plotly_chart(fig_pop, use_container_width=True)
        
    with col2:
        # Value Chart
        val_counts = filtered_df.groupby('Cluster')['Total_Lifetime_Spend'].sum().reset_index()
        fig_val = px.bar(val_counts, x='Cluster', y='Total_Lifetime_Spend', title='Total Revenue by Segment')
        st.plotly_chart(fig_val, use_container_width=True)

with tab2:
    st.header("Identify At-Risk High Rollers")
    st.markdown("Use the sliders to identify valuable customers who are breaking their buying rhythm.")
    
    # Interactive thresholds
    risk_threshold = st.slider("Minimum Churn Risk Index", 0.0, 3.0, 1.2)
    spend_threshold = st.slider("Minimum Lifetime Spend ($)", 0, int(df['Total_Lifetime_Spend'].max()), 100)
    
    # Plotly Interactive Scatter
    target_df = filtered_df[(filtered_df['Churn_Risk_Index_CRI'] >= risk_threshold) & 
                            (filtered_df['Total_Lifetime_Spend'] >= spend_threshold)]
    
    fig_scatter = px.scatter(
        target_df, 
        x='Churn_Risk_Index_CRI', 
        y='Average_Order_Value_AOV', 
        color='Cluster',
        size='Total_Lifetime_Spend',
        hover_data=['customer_id', 'Tenure_Days'],
        title="High Value / High Risk Matrix"
    )
    # Add a visual danger zone line
    fig_scatter.add_vline(x=1.5, line_dash="dash", line_color="red", annotation_text="Critical Churn Risk")
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.header("Export Target Lists")
    st.markdown(f"Found **{len(target_df)}** customers matching your Risk/Value criteria.")
    
    # Display actionable table
    st.dataframe(target_df[['customer_id', 'Cluster', 'Churn_Risk_Index_CRI', 'Total_Lifetime_Spend', 'Primary_Category_Name']])
    
    # Native Streamlit CSV Downloader
    csv = target_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Segment for CRM Activation",
        data=csv,
        file_name='targeted_at_risk_customers.csv',
        mime='text/csv',
    )