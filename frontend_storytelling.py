"""Customer-level presentation helpers; no model fitting or transaction processing."""
import numpy as np
import pandas as pd

METRICS = {
    "Total_Lifetime_Spend": "Observed spend", "Average_Order_Value_AOV": "AOV",
    "Recency_Days": "Recency (days)", "Total_Lifetime_Transactions": "Transactions",
    "Category_Width": "Categories", "Average_InterPurchase_Time_AIPT": "Purchase interval (days)",
    "Churn_Risk_Index_CRI": "Purchase delay", "Inferred_Discount_Affinity": "Discount affinity",
}
REQUIRED = {"customer_id", "Cluster", "Total_Lifetime_Spend", "Average_Order_Value_AOV",
            "Recency_Days", "Average_InterPurchase_Time_AIPT", "Churn_Risk_Index_CRI"}


def prepare_customers(frame):
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError("Missing customer columns: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    if frame.empty or frame['customer_id'].isna().any():
        raise ValueError('Expected nonempty customer data with non-null IDs.')
    frame['customer_id'] = frame['customer_id'].astype(str)
    if frame['customer_id'].duplicated().any() or frame['Cluster'].isna().any():
        raise ValueError('Expected one row per customer and non-null cluster IDs.')
    return frame


def segment_profiles(customers):
    profiles = customers.groupby('Cluster', sort=True).agg(
        customers=('customer_id', 'size'), spend=('Total_Lifetime_Spend', 'sum'),
        median_spend=('Total_Lifetime_Spend', 'median'),
        median_aov=('Average_Order_Value_AOV', 'median'), median_recency=('Recency_Days', 'median'))
    repeat = customers[customers['Average_InterPurchase_Time_AIPT'].gt(0)]
    profiles['median_delay'] = repeat.groupby('Cluster')['Churn_Risk_Index_CRI'].median()
    profiles['repeat_share'] = repeat.groupby('Cluster').size().reindex(profiles.index, fill_value=0) / profiles['customers']
    profiles['customer_share'] = profiles['customers'] / len(customers) * 100
    total = customers['Total_Lifetime_Spend'].sum()
    profiles['spend_share'] = profiles['spend'] / total * 100 if total > 0 else 0.0
    low, high = customers['Total_Lifetime_Spend'].quantile([.25, .75])
    labels, actions = [], []
    for cluster, row in profiles.iterrows():
        value = 'high-spend' if row.median_spend >= high else 'low-spend' if row.median_spend <= low else 'mid-spend'
        if row.repeat_share < .5 or pd.isna(row.median_delay):
            behavior, action = 'limited repeat history', 'Test second-purchase messaging'
        elif row.median_delay >= 2:
            behavior, action = 'overdue repeat buyers', 'Test reactivation outreach'
        elif row.median_delay > 1:
            behavior, action = 'delayed repeat buyers', 'Test purchase reminders'
        else:
            behavior, action = 'on-cadence repeat buyers', 'Test loyalty or cross-sell messaging'
        labels.append(f'Cluster {cluster} · {value} {behavior}')
        actions.append(action)
    profiles['label'], profiles['action_hypothesis'] = labels, actions
    return profiles.sort_values('median_spend', ascending=False, kind='stable')


def profile_matrix(customers, cluster_order):
    metrics = [m for m in METRICS if m in customers.columns]
    means = customers.groupby('Cluster')[metrics].mean().reindex(cluster_order)
    overall, std = customers[metrics].mean(), customers[metrics].std().replace(0, np.nan)
    z = ((means - overall) / std).fillna(0.0)
    custom = np.stack([means.to_numpy(), np.broadcast_to(overall.to_numpy(), means.shape), z.to_numpy()], axis=-1)
    return metrics, z, custom


def selected_customer_ids(points):
    return {str(point['customdata'][0]) for point in points
            if point.get('customdata') is not None and len(point['customdata']) > 0}


def render_contribution(profiles, selected, colors):
    import plotly.express as px
    import streamlit as st
    visible = profiles.loc[profiles.index.isin(selected)]
    order = visible['label'].tolist()
    st.subheader('Which segments contribute most?')
    for column, metric, title in zip(st.columns(2), ['customer_share', 'spend_share'], ['Customer share', 'Observed-spend share']):
        fig = px.bar(visible, x=metric, y='label', color='label', orientation='h',
            color_discrete_map=colors, category_orders={'label': order}, text_auto='.1f',
            hover_data={'customers': True, 'spend': ':,.0f', 'label': False},
            labels={metric: '% of full base', 'label': ''}, title=title)
        fig.update_traces(texttemplate='%{x:.1f}%')
        fig.update_layout(showlegend=False, height=max(300, len(order) * 45), margin=dict(t=50, b=30))
        fig.update_xaxes(range=[0, 105])
        with column:
            st.plotly_chart(fig, use_container_width=True)
    leader = visible.sort_values('spend_share', ascending=False).iloc[0]
    st.info(f"{leader['label']} represents {leader['customer_share']:.1f}% of customers and {leader['spend_share']:.1f}% of observed spend.")
    st.caption('Full-base denominators; observed spend is historical, not predicted lifetime value. Spend shares are undefined for a non-positive base total; bars then display zero.')


def render_profiles(customers, profiles, selected):
    import plotly.express as px
    import streamlit as st
    visible = profiles.loc[profiles.index.isin(selected)]
    metrics, z, custom = profile_matrix(customers, visible.index.tolist())
    st.header('Segment profiles')
    fig = px.imshow(z.to_numpy(), x=[METRICS[m] for m in metrics],
        y=[f"{r['label']} (n={int(r['customers']):,})" for _, r in visible.iterrows()],
        color_continuous_scale='RdBu_r', color_continuous_midpoint=0, zmin=-3, zmax=3, text_auto='.1f', aspect='auto')
    fig.update_traces(customdata=custom, hovertemplate='%{y}<br>%{x}<br>Segment mean: %{customdata[0]:,.2f}<br>Full-base mean: %{customdata[1]:,.2f}<br>Standardized difference: %{customdata[2]:.2f}<extra></extra>')
    fig.update_layout(height=max(400, len(visible) * 55), coloraxis_colorbar_title='Std. difference')
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Red means above average, not necessarily better: higher recency and delay can signal overdue purchases. Values are (segment mean − full-base mean) / full-base standard deviation. Colors saturate at ±3; constant metrics display zero.')
    summary = visible[['label', 'customers', 'median_spend', 'median_delay', 'repeat_share', 'action_hypothesis']].reset_index(drop=True)
    st.dataframe(summary, use_container_width=True, column_config={
        'label': 'Segment', 'customers': 'Customers',
        'median_spend': st.column_config.NumberColumn('Median observed spend', format='$%.2f'),
        'median_delay': st.column_config.NumberColumn('Repeat-buyer median delay', format='%.2f'),
        'repeat_share': st.column_config.NumberColumn('Repeat-history fraction', format='%.2f', help='0.50 means 50% with positive purchase intervals.'),
        'action_hypothesis': 'Action hypothesis',
    })
    st.caption('Descriptions use full-base spend quartiles and repeat-buyer cadence. Actions are hypotheses, not validated recommendations; delay is not a churn probability.')


def render_risk(customers, selected, colors, order, source_fingerprint):
    import hashlib
    import inspect
    import plotly.express as px
    import streamlit as st
    st.header('Risk–value portfolio')
    st.caption('Purchase delay = recency / average purchase interval. Select points to inspect customers; it is not a calibrated churn probability.')
    view = customers[customers['Cluster'].isin(selected)]
    percentile = st.slider('High-value spend percentile', 50, 99, 80, key='risk_percentile')
    log = st.checkbox('Logarithmic spend axis', value=True, key='risk_log')
    show_history = st.checkbox('Show insufficient repeat history', value=True, key='risk_history')
    repeat = customers[customers['Average_InterPurchase_Time_AIPT'].gt(0)]
    threshold = repeat['Total_Lifetime_Spend'].quantile(percentile / 100)
    plot = view.copy()
    plot['History'] = np.where(plot['Average_InterPurchase_Time_AIPT'].gt(0), 'Repeat history', 'Insufficient history')
    plot.loc[plot['History'].eq('Insufficient history'), 'Churn_Risk_Index_CRI'] = 0
    if not show_history:
        plot = plot[plot['History'].eq('Repeat history')]
    if log:
        omitted = int(plot['Total_Lifetime_Spend'].le(0).sum())
        plot = plot[plot['Total_Lifetime_Spend'].gt(0)]
        if omitted:
            st.caption(f'{omitted:,} non-positive-spend customers omitted from the log chart, not from summaries.')
    ids = set()
    if not plot.empty:
        fig = px.scatter(plot, x='Churn_Risk_Index_CRI', y='Total_Lifetime_Spend', color='Segment', symbol='History',
            symbol_map={'Repeat history': 'circle', 'Insufficient history': 'x'}, color_discrete_map=colors,
            category_orders={'Segment': order}, custom_data=['customer_id'], hover_name='customer_id',
            hover_data={'Recency_Days': True, 'Average_Order_Value_AOV': ':,.2f', 'Average_InterPurchase_Time_AIPT': ':.1f', 'Total_Lifetime_Spend': ':,.2f'},
            labels={'Churn_Risk_Index_CRI': 'Purchase delay', 'Total_Lifetime_Spend': 'Observed spend ($)'}, log_y=log, opacity=.65)
        for x, title in [(1, 'On cadence'), (2, '2× cadence'), (3, '3× cadence')]:
            fig.add_vline(x=x, line_dash='dot', annotation_text=title)
        if np.isfinite(threshold) and (not log or threshold > 0):
            fig.add_hline(y=threshold, line_dash='dot', annotation_text=f'Full-base repeat-buyer P{percentile}: ${threshold:,.0f}')
        fig.update_layout(height=550, dragmode='lasso')
        signature = f'{source_fingerprint}|{selected}|{percentile}|{log}|{show_history}'
        key = 'risk_' + hashlib.sha256(signature.encode()).hexdigest()[:12]
        if 'on_select' in inspect.signature(st.plotly_chart).parameters:
            event = st.plotly_chart(fig, use_container_width=True, key=key, on_select='rerun', selection_mode=('points', 'box', 'lasso'))
            ids = selected_customer_ids(event.selection.points)
        else:
            st.plotly_chart(fig, use_container_width=True)
            st.caption('Upgrade to Streamlit >=1.35 for selection; default targets are shown below.')
    else:
        st.info('No plottable customers with these settings. Change the log/history options.')
    st.caption('Crosses at x=0 mean insufficient repeat history; zero is only a plotting convention, not a delay estimate.')
    if ids:
        targets = view[view['customer_id'].isin(ids)].copy()
        st.subheader(f'Selected customers: {len(targets):,}')
        st.caption(f"Observed spend: ${targets['Total_Lifetime_Spend'].sum():,.0f}")
    else:
        targets = view[view['Average_InterPurchase_Time_AIPT'].gt(0) & view['Churn_Risk_Index_CRI'].ge(2) & view['Total_Lifetime_Spend'].ge(threshold)].nlargest(20, 'Total_Lifetime_Spend')
        st.subheader('Top targets: high observed spend and purchase delay')
        st.caption('Default: up to 20 repeat buyers with delay ≥2 and spend at or above the full-base repeat-buyer threshold.')
    if targets.empty:
        st.info('No customers meet these criteria.')
    else:
        columns = ['customer_id', 'Segment', 'Total_Lifetime_Spend', 'Churn_Risk_Index_CRI', 'Recency_Days', 'Average_Order_Value_AOV']
        st.dataframe(targets[columns].reset_index(drop=True), use_container_width=True, column_config={
            'Total_Lifetime_Spend': st.column_config.NumberColumn('Observed spend', format='$%.2f'),
            'Average_Order_Value_AOV': st.column_config.NumberColumn('AOV', format='$%.2f'),
            'Churn_Risk_Index_CRI': st.column_config.NumberColumn('Purchase delay', format='%.2f'),
        })


def render_radar(customers, selected, colors, order):
    import plotly.graph_objects as go
    import streamlit as st
    metrics = [m for m in METRICS if m in customers.columns]
    means = customers[customers['Cluster'].isin(selected)].groupby('Segment')[metrics].mean().reindex(order)
    normalized = ((means - means.min()) / (means.max() - means.min()).replace(0, np.nan)).fillna(.5)
    with st.expander('Radar view: normalized within selection'):
        st.caption('Shape depends on the selection; use the heatmap for stable full-base comparison.')
        fig = go.Figure()
        for label, row in normalized.iterrows():
            values, names = row.tolist(), [METRICS[m] for m in metrics]
            fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=names + names[:1], fill='toself', name=label, line_color=colors[label]))
        fig.update_layout(height=450, polar_radialaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)
