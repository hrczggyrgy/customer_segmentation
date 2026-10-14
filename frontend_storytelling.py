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

# Feature-specific wording, based on the definitions in segmentation.py.
# 'format' renders raw averages with units; 'definition' says how the feature is built;
# 'above'/'below' describe the direction without implying better or worse performance.
METRIC_NARRATIVES = {
    'Total_Lifetime_Spend': dict(
        format='${:,.2f}',
        definition='Sum of quantity x price over observed purchase lines.',
        above='Higher historical spend in the observed period; not predicted lifetime value.',
        below='Lower historical spend in the observed period; not predicted lifetime value.'),
    'Average_Order_Value_AOV': dict(
        format='${:,.2f}',
        definition='Observed spend divided by distinct transaction IDs.',
        above='Larger average spend per transaction.',
        below='Smaller average spend per transaction.'),
    'Recency_Days': dict(
        format='{:,.1f} days',
        definition='Days from the last observed purchase to the day after the latest date in the data.',
        above='Longer since the last observed purchase; a timing description, not a verdict on loyalty.',
        below='More recent last observed purchase.'),
    'Total_Lifetime_Transactions': dict(
        format='{:,.2f} transactions',
        definition='Distinct transaction IDs observed per customer.',
        above='More distinct transactions in the observed period.',
        below='Fewer distinct transactions in the observed period.'),
    'Category_Width': dict(
        format='{:,.2f} categories',
        definition='Distinct product categories purchased.',
        above='Purchases span more product categories.',
        below='Purchases are concentrated in fewer product categories.'),
    'Average_InterPurchase_Time_AIPT': dict(
        format='{:,.1f} days',
        definition='Mean gap between distinct purchase dates; zero with fewer than two purchase dates.',
        above='Longer gaps between purchase dates.',
        below='Shorter gaps between purchase dates, or zeros from single-date customers; check repeat-history coverage.'),
    'Churn_Risk_Index_CRI': dict(
        format='{:,.2f}x',
        definition='Recency divided by average purchase interval when that interval is positive; otherwise zero.',
        above='Recency is large relative to typical purchase gaps; a descriptive delay ratio, not a churn probability.',
        below='Lower delay ratio; zero for single-date customers does not indicate healthy behavior.'),
    'Inferred_Discount_Affinity': dict(
        format='{:,.3f} lines/unit',
        definition='Lines priced below 80% of the SKU maximum observed price, divided by purchased units.',
        above='More below-80%-of-maximum-price lines per unit; not a discount percentage or share of transactions.',
        below='Fewer below-80%-of-maximum-price lines per unit; not a discount percentage or share of transactions.'),
}


def format_metric_value(metric, value):
    """Format a raw average with its unit."""
    spec = METRIC_NARRATIVES.get(metric)
    return spec['format'].format(value) if spec else f'{value:,.2f}'


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


def contribution_metrics(customer_share, spend_share, total_spend):
    """Return spend/customer-share ratio and percentage-point gap, or N/A values."""
    try:
        customer_share, spend_share, total_spend = map(float, (customer_share, spend_share, total_spend))
    except (TypeError, ValueError):
        return None, None
    if (not np.isfinite([customer_share, spend_share, total_spend]).all()
            or customer_share <= 0 or spend_share < 0 or total_spend <= 0):
        return None, None
    ratio, gap = spend_share / customer_share, spend_share - customer_share
    return (ratio, gap) if np.isfinite([ratio, gap]).all() else (None, None)


def segment_differentiators(customers, cluster_order, max_traits=3, min_abs_z=0.25):
    """Rank descriptive full-base standardized differences; not significance tests."""
    if not isinstance(max_traits, int) or max_traits < 1:
        raise ValueError('max_traits must be a positive integer.')
    if not np.isfinite(min_abs_z) or min_abs_z < 0:
        raise ValueError('min_abs_z must be finite and non-negative.')
    metrics, z, custom = profile_matrix(customers, cluster_order)
    stds = customers[metrics].std()
    result = {}
    for i, cluster in enumerate(z.index):
        eligible = {}
        for j, metric in enumerate(metrics):
            raw, overall, score = custom[i, j]
            std = stds[metric]
            if (np.isfinite([raw, overall, score, std]).all() and std > 0
                    and score != 0 and abs(score) >= min_abs_z):
                eligible[metric] = score
        ranking = pd.Series(eligible, dtype=float).abs().nlargest(max_traits)
        traits = []
        for metric in ranking.index:
            j = metrics.index(metric)
            raw, overall, score = custom[i, j]
            direction = 'above' if score > 0 else 'below'
            spec = METRIC_NARRATIVES.get(metric, {})
            context = spec.get('above' if score > 0 else 'below', '')
            traits.append(dict(metric=metric, label=METRICS[metric], z_score=float(score),
                direction=direction, segment_mean=float(raw), overall_mean=float(overall),
                context=context, definition=spec.get('definition', '')))
        result[cluster] = traits
    return result


def render_contribution(profiles, selected, colors):
    import plotly.express as px
    import streamlit as st
    visible = profiles.loc[profiles.index.isin(selected)]
    
    # Create short labels for charts: "C2 · High-spend, overdue"
    def short_label(full_label: str) -> str:
        # "Cluster 2 · high-spend overdue repeat buyers" -> "C2 · High-spend, overdue"
        parts = full_label.split(' · ')
        if len(parts) != 2:
            return full_label
        cluster_part = parts[0].replace('Cluster ', 'C')
        desc = parts[1]
        # Simplify description
        desc = desc.replace('repeat buyers', '').replace('repeat history', 'history')
        desc = desc.replace('high-spend', 'High-spend').replace('mid-spend', 'Mid-spend').replace('low-spend', 'Low-spend')
        desc = desc.replace('on-cadence', 'on-cadence').replace('overdue', 'overdue').replace('delayed', 'delayed').replace('limited', 'limited')
        desc = desc.strip()
        return f"{cluster_part} · {desc}"
    
    visible = visible.copy()
    visible['short_label'] = visible['label'].apply(short_label)
    order = visible['short_label'].tolist()
    label_to_full = dict(zip(visible['short_label'], visible['label']))
    
    st.subheader('Which segments contribute most?')
    
    # Headline: leading segment punchline
    leader = visible.sort_values('spend_share', ascending=False).iloc[0]
    ratio, gap = contribution_metrics(leader['customer_share'], leader['spend_share'], profiles['spend'].sum())
    if ratio is not None and gap is not None:
        headline = f"{label_to_full[leader['short_label']].split(' · ')[0]} is {leader['customer_share']:.1f}% of customers but {leader['spend_share']:.1f}% of spend (gap: {gap:+.1f} pp, index: {ratio:.2f}×)"
        st.info(headline)
    
    # Single grouped horizontal bar chart
    share_plot = visible[['short_label', 'customer_share', 'spend_share']].melt(
        id_vars='short_label',
        var_name='Share type',
        value_name='Share (%)'
    )
    share_plot['Share type'] = share_plot['Share type'].map({
        'customer_share': 'Customer share',
        'spend_share': 'Observed-spend share',
    })
    
    fig = px.bar(
        share_plot,
        x='Share (%)',
        y='short_label',
        color='Share type',
        orientation='h',
        barmode='group',
        text_auto='.1f',
        labels={'short_label': 'Segment', 'Share (%)': '% of full base'},
        category_orders={'short_label': order, 'Share type': ['Customer share', 'Observed-spend share']},
        color_discrete_map={
            'Customer share': '#636EFA',
            'Observed-spend share': '#EF553B',
        },
        custom_data=[share_plot['short_label'].map(visible.set_index('short_label')['customers'])],
    )
    fig.update_traces(
        texttemplate='%{x:.1f}%',
        hovertemplate='Segment: %{y}<br>Share type: %{fullData.name}<br>Share: %{x:.1f}%<br>Customers: %{customdata[0]:,}<extra></extra>',
    )
    fig.update_layout(
        barmode='group',
        height=max(300, len(order) * 45),
        margin=dict(t=30, b=30),
        showlegend=True,
        legend_title_text='',
        yaxis=dict(automargin=True),
    )
    fig.update_xaxes(range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)
    
    # KPI cards
    ratio_col, gap_col = st.columns(2)
    ratio_col.metric('Spend contribution index', f'{ratio:.2f}×' if ratio is not None else 'N/A')
    gap_col.metric('Gap vs. customer share', f'{gap:+.1f} pp' if gap is not None else 'N/A')
    st.caption('Leading segment = highest observed-spend share among selected segments. Denominators use the full base. Index = spend share / customer share; gap = spend share − customer share in percentage points, not change over time. Observed spend is historical, not predicted lifetime value.')


def _card():
    import streamlit as st
    # Fallback for older Streamlit versions without st.container
    if hasattr(st, 'container'):
        try:
            return st.container(border=True)
        except TypeError:
            return st.container()
    # Fallback: just return a context manager that does nothing
    from contextlib import contextmanager
    @contextmanager
    def dummy_container():
        yield None
    return dummy_container()


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
    st.subheader('What distinguishes these segments?')
    differentiators = segment_differentiators(customers, visible.index.tolist())
    columns = st.columns(2)
    for i, (cluster, row) in enumerate(visible.iterrows()):
        with columns[i % 2]:
            with _card():
                st.markdown(f"**{row['label']}**")
                traits = differentiators[cluster]
                if not traits:
                    st.caption('No eligible differences at or above 0.25 full-base standard deviations.')
                for trait in traits:
                    arrow = '▲' if trait['direction'] == 'above' else '▼'
                    st.markdown(f"{arrow} **{trait['label']}** · {abs(trait['z_score']):.2f} SD {trait['direction']} the full-base average")
                    st.caption(f"Segment mean {format_metric_value(trait['metric'], trait['segment_mean'])} vs. full-base mean "
                        f"{format_metric_value(trait['metric'], trait['overall_mean'])}. {trait['context']} Definition: {trait['definition']}")
    st.caption('Up to three traits ranked by absolute standardized difference; 0.25 SD is a display threshold, not statistical significance. Differences are descriptive, not causal or proof of better performance. Correlated traits are not independent evidence. Cadence/delay metrics include zeros for insufficient repeat history; inspect the repeat-history fraction below.')
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
    import plotly.graph_objects as go
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
        # WebGL rendering for responsiveness
        fig = px.scatter(plot, x='Churn_Risk_Index_CRI', y='Total_Lifetime_Spend', color='Segment', symbol='History',
            symbol_map={'Repeat history': 'circle', 'Insufficient history': 'x'}, color_discrete_map=colors,
            category_orders={'Segment': order}, custom_data=['customer_id'], hover_name='customer_id',
            hover_data={'Recency_Days': True, 'Average_Order_Value_AOV': ':,.2f', 'Average_InterPurchase_Time_AIPT': ':.1f', 'Total_Lifetime_Spend': ':,.2f'},
            labels={'Churn_Risk_Index_CRI': 'Purchase delay', 'Total_Lifetime_Spend': 'Observed spend ($)'}, log_y=log, opacity=.65,
            render_mode='webgl')
        for x, title in [(1, 'On cadence'), (2, '2× cadence'), (3, '3× cadence')]:
            fig.add_vline(x=x, line_dash='dot', annotation_text=title)
        if np.isfinite(threshold) and (not log or threshold > 0):
            fig.add_hline(y=threshold, line_dash='dot', annotation_text=f'Full-base repeat-buyer P{percentile}: ${threshold:,.0f}')
        
        # Priority zone: high spend + delay >= 2
        if np.isfinite(threshold):
            x_max = plot['Churn_Risk_Index_CRI'].max() if not plot.empty else 5
            x_max = max(x_max, 3)
            y_max = plot['Total_Lifetime_Spend'].max() if not plot.empty else threshold * 10
            if log:
                y_max = y_max if np.isfinite(y_max) and y_max > threshold else threshold * 10
            
            fig.add_shape(
                type='rect',
                x0=2, x1=x_max, y0=threshold, y1=y_max,
                fillcolor='rgba(239, 85, 59, 0.08)',
                line=dict(width=1, color='#EF553B', dash='dash'),
                layer='below',
            )
            fig.add_annotation(
                x=2.1, y=threshold * (1.2 if not log else 1.5),
                text='Priority outreach zone',
                showarrow=False,
                font=dict(size=11, color='#EF553B'),
                xanchor='left', yanchor='bottom',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#EF553B',
                borderwidth=1,
            )
        
        fig.update_layout(height=550, dragmode='lasso', yaxis=dict(automargin=True))
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
        # Download button
        csv = targets[columns].to_csv(index=False).encode('utf-8')
        st.download_button(
            'Download targets (CSV)',
            data=csv,
            file_name='risk_value_targets.csv',
            mime='text/csv',
        )


def render_zscore_small_multiples(customers, profiles, selected, colors, order):
    """Show standardized differences (z-scores) as small-multiple horizontal bar charts.

    Each facet is one segment; x-axis is z-score with zero line.
    This is stable across selections because it uses full-base standardization.
    """
    import plotly.express as px
    import streamlit as st
    visible = profiles.loc[profiles.index.isin(selected)]
    if visible.empty:
        return
    metrics, z, _ = profile_matrix(customers, visible.index.tolist())
    z.index = visible.loc[z.index, 'label'].values
    
    # Melt for faceting
    z_long = z.reset_index().melt(id_vars='index', var_name='metric', value_name='z')
    z_long = z_long.rename(columns={'index': 'Segment'})
    z_long['metric_label'] = z_long['metric'].map(METRICS)
    
    # Create short labels for facets
    def short_label(full_label: str) -> str:
        parts = full_label.split(' · ')
        if len(parts) != 2:
            return full_label
        cluster_part = parts[0].replace('Cluster ', 'C')
        desc = parts[1].replace('repeat buyers', '').replace('repeat history', 'history')
        desc = desc.replace('high-spend', 'High-spend').replace('mid-spend', 'Mid-spend').replace('low-spend', 'Low-spend')
        desc = desc.strip()
        return f"{cluster_part} · {desc}"
    
    z_long['Segment'] = z_long['Segment'].apply(short_label)
    
    st.subheader('Segment profiles (standardized differences)')
    st.caption('Each chart shows how a segment differs from the full-base mean in standard deviations. Red = above average (not necessarily better).')
    
    fig = px.bar(
        z_long,
        x='z', y='metric_label', color='Segment',
        facet_col='Segment', facet_col_wrap=3,
        color_discrete_map={short_label(label): colors[label] for label in order},
        labels={'z': 'Standardized difference (z)', 'metric_label': ''},
        height=max(400, len(metrics) * 35 + 100),
    )
    # Add zero line to each facet
    fig.add_vline(x=0, line_color='#64748B', line_width=1, line_dash='solid')
    fig.update_yaxes(autorange='reversed', automargin=True)
    fig.update_xaxes(range=[-3.5, 3.5], dtick=1)
    # Clean facet titles
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


def render_radar(customers, selected, colors, order):
    """Deprecated: kept for backward compatibility. Use render_zscore_small_multiples instead."""
    # Radar chart removed; small multiples are more honest (stable z-scores vs. selection-dependent min-max)
    pass
