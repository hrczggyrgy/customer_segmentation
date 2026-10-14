import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def plot_cohort_retention(transactions_df):
    """Generates a Triangular Cohort Retention Heatmap."""
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
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0.0, vmax=0.5)
    plt.title('Customer Cohort Retention Matrix', fontsize=16)
    plt.ylabel('Acquisition Cohort')
    plt.xlabel('Months Since First Purchase')
    plt.tight_layout()
    plt.show()

def plot_customer_journey_sankey(transactions_df):
    """Generates a Flowing Sankey Diagram of the First 3 Purchases."""
    df = transactions_df.copy()
    
    # Sort transactions chronologically per customer
    df = df.sort_values(['customer_id', 'transaction_date'])
    
    # Assign a purchase sequence number (1st purchase, 2nd purchase, etc.)
    df['Purchase_Seq'] = df.groupby('customer_id').cumcount() + 1
    
    # Isolate the first 3 categories a customer bought
    journey = df[df['Purchase_Seq'] <= 3].pivot(index='customer_id', columns='Purchase_Seq', values='category')
    journey.columns = ['Step_1', 'Step_2', 'Step_3']
    journey = journey.dropna(subset=['Step_1', 'Step_2']) # Must have at least 2 purchases
    
    # Build source/target pairs for Step 1 -> Step 2
    step1_2 = journey.groupby(['Step_1', 'Step_2']).size().reset_index(name='Count')
    step1_2['Source'] = step1_2['Step_1'] + " (1st)"
    step1_2['Target'] = step1_2['Step_2'] + " (2nd)"
    
    # Build source/target pairs for Step 2 -> Step 3
    step2_3 = journey.dropna(subset=['Step_3']).groupby(['Step_2', 'Step_3']).size().reset_index(name='Count')
    step2_3['Source'] = step2_3['Step_2'] + " (2nd)"
    step2_3['Target'] = step2_3['Step_3'] + " (3rd)"
    
    # Combine links
    links = pd.concat([step1_2[['Source', 'Target', 'Count']], step2_3[['Source', 'Target', 'Count']]])
    
    # Map node strings to integer IDs required by Plotly
    unique_nodes = list(pd.concat([links['Source'], links['Target']]).unique())
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}
    
    links['Source_ID'] = links['Source'].map(node_mapping)
    links['Target_ID'] = links['Target'].map(node_mapping)
    
    # Plotly Sankey
    fig = go.Figure(data=[go.Sankey(
        node={
            'pad': 20, 'thickness': 30, 'line': {'color': "black", 'width': 0.5},
            'label': unique_nodes, 'color': "lightblue"
        },
        link={
            'source': links['Source_ID'], 'target': links['Target_ID'], 'value': links['Count'],
            'color': "rgba(173, 216, 230, 0.4)"
        }
    )])
    
    fig.update_layout(title_text="Customer Journey: Category Migration (First 3 Purchases)", font_size=12)
    fig.show()