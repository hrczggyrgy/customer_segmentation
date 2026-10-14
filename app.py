import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from datetime import timedelta
from segmentation import AdvancedCustomerSegmentation
from data_safety import source_fingerprint, generate_session_segments, finalized_month_end, month_end, retention_matrix
from frontend_storytelling import (
    METRICS, prepare_customers, segment_profiles, render_contribution,
    render_profiles, render_risk, render_radar,
)

st.set_page_config(page_title="Advanced Customer Analytics", layout="wide")
st.title("Customer Intelligence Platform")
st.caption("Understand segment value, compare behavior, and explore customer-level action hypotheses.")

for key, default in {
    "segments_df": None, "tx_df": None, "segments_generated": False,
    "tx_fingerprint": None, "segments_fingerprint": None, "pca_metadata": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def pca_variance_by_component(metadata, fingerprint):
    '''Explained-variance ratios by PC name, only from valid, fingerprint-matched metadata.'''
    if not isinstance(metadata, dict) or metadata.get('source_fingerprint') != fingerprint:
        return {}
    try:
        ratios = np.asarray(metadata.get('explained_variance_ratio'), dtype=float)
    except (TypeError, ValueError):
        return {}
    if (ratios.ndim != 1 or ratios.size == 0 or not np.isfinite(ratios).all()
            or (ratios < 0).any() or ratios.sum() > 1 + 1e-6):
        return {}
    return {f'PC{i + 1}': float(r) for i, r in enumerate(ratios)}


def pc_axis_label(pc, variance):
    return f'{pc} ({variance[pc]:.1%} of variance)' if pc in variance else pc


def cohort_sizes(lazy_transactions):
    '''Distinct customers per first-purchase month, from the full transaction base.'''
    sizes = (lazy_transactions.group_by('customer_id')
        .agg(pl.col('transaction_date').dt.truncate('1mo').min().alias('CohortMonth'))
        .group_by('CohortMonth').agg(pl.col('customer_id').n_unique().alias('size')).collect())
    return {month.strftime('%Y-%m'): int(n) for month, n in sizes.iter_rows()}


def get_or_generate_segments(tx_df, tx_fingerprint, force_refresh=False):
    if tx_df is None:
        st.error('Upload transactions first.')
        st.stop()
    try:
        with st.sidebar.status('Preparing session-only segments...', expanded=True) as status:
            result = generate_session_segments(
                st.session_state, tx_df, tx_fingerprint,
                lambda: AdvancedCustomerSegmentation(max_clusters=8, n_components=6),
                force=force_refresh,
            )
            status.update(label='Segments ready in this session', state='complete')
        return result
    except Exception as error:
        st.error(f'Segmentation failed: {error}')
        st.stop()


if st.session_state.tx_fingerprint and len(st.session_state.tx_fingerprint) != 64:
    st.session_state.update(tx_df=None, tx_fingerprint=None, segments_df=None,
        segments_fingerprint=None, segments_generated=False, pca_metadata=None,
        show_retention=False, retention_key=None, retention_result=None)

st.sidebar.header("Data upload")
uploaded_transactions = st.sidebar.file_uploader(
    "Cleaned transactions (Parquet)", type=["parquet"],
    help="Uploads and generated results stay in this session; the app does not save upload artifacts to disk.",
)
if uploaded_transactions is not None:
    new_fp = source_fingerprint(uploaded_transactions)
    if st.session_state.tx_fingerprint != new_fp:
        st.session_state.tx_df = pl.read_parquet(io.BytesIO(uploaded_transactions.getvalue()))
        st.session_state.tx_fingerprint = new_fp
        st.session_state.segments_generated = False
        st.session_state.segments_df = None
        st.session_state.segments_fingerprint = None
        st.session_state.pca_metadata = None
        st.session_state.show_retention = False
        st.session_state.retention_key = None
        st.session_state.retention_result = None
        st.sidebar.info("New data detected. Generate segments below.")
elif st.session_state.tx_df is None:
    try:
        st.session_state.tx_df = pl.read_parquet("customer_transactions_clean.parquet")
        st.session_state.tx_fingerprint = source_fingerprint("customer_transactions_clean.parquet")
    except FileNotFoundError:
        st.info("Upload cleaned transactions from the sidebar to begin.")
        st.stop()
tx_df = st.session_state.tx_df
tx_lazy = tx_df.lazy()

st.sidebar.header("Segmentation controls")
ready = st.session_state.segments_generated and st.session_state.segments_df is not None
refresh = st.sidebar.button("Refresh segments" if ready else "Generate segments", type="primary", use_container_width=True)
force = st.sidebar.button("Force regenerate", help="Recompute results for this session without clearing other visitors' results.", use_container_width=True)
if refresh or force:
    with st.spinner("Preparing customer segments..."):
        segments, metadata = get_or_generate_segments(tx_df, st.session_state.tx_fingerprint, force_refresh=force)
        st.session_state.segments_df = segments
        st.session_state.segments_fingerprint = st.session_state.tx_fingerprint
        st.session_state.pca_metadata = metadata
        st.session_state.segments_generated = True
    st.rerun()
if not ready:
    st.info("Generate segments using the sidebar to open the dashboard.")
    st.stop()
if st.session_state.segments_fingerprint != st.session_state.tx_fingerprint:
    st.warning("Segments do not match the current transactions. Refresh them first.")
    st.stop()
segments_df = st.session_state.segments_df
if 'Primary_Category_Name' not in segments_df.columns:
    primary_cat = tx_lazy.group_by('customer_id').agg(
        pl.col('category').mode().first().alias('Primary_Category_Name')
    ).collect()
    segments_df = segments_df.join(primary_cat, on='customer_id', how='left').with_columns(
        pl.col('Primary_Category_Name').fill_null('Unknown')
    )
    st.session_state.segments_df = segments_df

try:
    customers = prepare_customers(segments_df.to_pandas())
except ValueError as error:
    st.error(str(error))
    st.stop()
profiles = segment_profiles(customers)
labels = profiles['label'].to_dict()
label_to_cluster = {label: cluster for cluster, label in labels.items()}
clusters = sorted(customers['Cluster'].unique().tolist())
colors = {labels[c]: px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)] for i, c in enumerate(clusters)}
customers['Segment'] = customers['Cluster'].map(labels)
selected = st.sidebar.multiselect('Global segment filters', clusters, default=clusters, format_func=lambda c: labels[c])
if not selected:
    st.info('Select at least one segment to explore customer behavior.')
    st.stop()
view = customers[customers['Cluster'].isin(selected)]
order = profiles.loc[profiles.index.isin(selected), 'label'].tolist()
category_orders = {'Segment': order}
min_date, max_date = tx_df['transaction_date'].min(), tx_df['transaction_date'].max()
st.caption(f"Observed transactions: {min_date.date()} to {max_date.date()} | Recency reference: {(max_date + timedelta(days=1)).date()}")
st.caption('Filters apply across customer views. Benchmarks and share denominators use the full base. Retention is labeled separately.')
for column, title, value in zip(st.columns(4),
    ['Selected customers', 'Observed spend', 'Median AOV', 'Median recency (days)'],
    [f"{len(view):,}", f"${view['Total_Lifetime_Spend'].sum():,.0f}", f"${view['Average_Order_Value_AOV'].median():,.2f}", f"{view['Recency_Days'].median():.0f}"]):
    column.metric(title, value)

render_contribution(profiles, selected, colors)
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Risk–Value Portfolio', 'Segment Profiles', 'Distributions', 'Model Exploration', 'Retention'])
with tab1:
    render_risk(customers, selected, colors, order, st.session_state.tx_fingerprint)
with tab2:
    render_profiles(customers, profiles, selected)
    render_radar(customers, selected, colors, order)
with tab3:
    st.header('Distributions within segments')
    st.caption('Segment averages hide variation. Compare distributions before interpreting a persona.')
    metrics = [m for m in METRICS if m in view.columns]
    metric = st.selectbox('Metric', metrics, format_func=lambda m: METRICS[m], key='distribution_metric')
    plot_type = st.radio('Plot type', ['Boxplot', 'Violin plot'], index=0, horizontal=True, key='distribution_plot_type')
    log = st.checkbox('Logarithmic axis', value=metric in ['Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Total_Lifetime_Transactions'], key=f'distribution_log_{metric}')
    chart_name = 'violin plot' if plot_type == 'Violin plot' else 'boxplot'
    finite = view[view[metric].replace([np.inf, -np.inf], np.nan).notna()]
    dist = finite[finite[metric].gt(0)] if log else finite
    if len(finite) < len(view):
        st.caption(f'{len(view) - len(finite):,} missing or non-finite values are not drawn; summary statistics ignore missing values.')
    if log and len(dist) < len(finite):
        st.caption(f'{len(finite) - len(dist):,} non-positive observations omitted from the logarithmic {chart_name}; summary statistics include them.')
    if dist.empty:
        st.info('No plottable observations for this metric and axis setting.')
    else:
        plot_options = dict(x='Segment', y=metric, color='Segment', color_discrete_map=colors, category_orders=category_orders, points='outliers', log_y=log, labels={metric: METRICS[metric]})
        fig = px.violin(dist, box=True, **plot_options) if plot_type == 'Violin plot' else px.box(dist, **plot_options)
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    stats = view.groupby('Segment')[metric].agg(['median', 'mean', 'std', 'count']).reindex(order)
    st.dataframe(stats.rename(columns={'median': 'Median', 'mean': 'Mean', 'std': 'Std dev', 'count': 'Customers'}), use_container_width=True)
with tab4:
    st.header('Model exploration: PCA')
    st.caption('Projections show only part of the clustering space; visual overlap does not establish full-space overlap.')
    if {'PC1', 'PC2', 'PC3'}.issubset(view.columns):
        variance = pca_variance_by_component(st.session_state.pca_metadata, st.session_state.tx_fingerprint)
        projection = st.radio('Projection', ['PC1 vs PC2', 'PC1 vs PC3', '3D (PC1/PC2/PC3)'], horizontal=True)
        is_3d = projection.startswith('3D')
        axes = ['PC1', 'PC2', 'PC3'] if is_3d else ['PC1', 'PC3' if projection.endswith('PC3') else 'PC2']
        axis_labels = {pc: pc_axis_label(pc, variance) for pc in axes}
        options = dict(color='Segment', color_discrete_map=colors, category_orders=category_orders, hover_name='customer_id', hover_data=['Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Primary_Category_Name'], opacity=.65, labels=axis_labels)
        fig = px.scatter_3d(view, x='PC1', y='PC2', z='PC3', **options) if is_3d else px.scatter(view, x=axes[0], y=axes[1], **options)
        centers = view.groupby('Segment')[axes].mean().reindex(order).dropna()
        for label, center in centers.iterrows():
            coords = dict(x=[center[axes[0]]], y=[center[axes[1]]])
            if is_3d:
                coords['z'] = [center[axes[2]]]
            trace = go.Scatter3d if is_3d else go.Scatter
            fig.add_trace(trace(**coords, mode='markers+text', text=[f'C{label_to_cluster[label]}'], textposition='top center',
                name=f'{label} center', showlegend=False, hovertext=[label],
                hovertemplate='%{hovertext}<br>Segment center: mean PC coordinates of displayed customers<extra></extra>',
                marker=dict(size=9 if is_3d else 14, color=colors[label], symbol='diamond', line=dict(color='black', width=2))))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        if variance:
            st.caption('Axis percentages are explained-variance ratios from fingerprint-matched PCA metadata. Diamonds labeled C# mark segment centers: mean PC coordinates of the displayed customers in each segment.')
        else:
            st.caption('Axis variance percentages are unavailable because PCA metadata is missing, invalid, or does not match these transactions. Diamonds labeled C# mark segment centers (mean PC coordinates).')
    else:
        st.info('PCA projections require PC1, PC2, and PC3 in the generated segments.')
    if st.checkbox('Show explained variance and loadings', value=True):
        metadata = st.session_state.pca_metadata
        if not metadata:
            st.info('Generate segments to populate PCA metadata.')
        elif metadata.get('source_fingerprint') != st.session_state.tx_fingerprint:
            st.warning('PCA metadata does not match these transactions. Refresh segments.')
        else:
            evr = metadata.get('explained_variance_ratio', [])
            evc = metadata.get('explained_variance_cumulative', [])
            if evr:
                pcs = [f'PC{i + 1}' for i in range(len(evr))]
                fig = go.Figure(go.Bar(x=pcs, y=np.array(evr) * 100, name='Individual'))
                if evc:
                    fig.add_trace(go.Scatter(x=[f'PC{i + 1}' for i in range(len(evc))], y=np.array(evc) * 100, name='Cumulative', mode='lines+markers', yaxis='y2'))
                    st.caption(f'Total explained variance ({len(evr)} components): {evc[-1]:.1%}')
                fig.update_layout(height=350, yaxis_title='Individual variance (%)', yaxis2=dict(title='Cumulative (%)', overlaying='y', side='right', range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True)
            loadings, names = metadata.get('loadings', []), metadata.get('feature_names', [])
            if loadings and names and len(loadings) == len(names):
                values = np.asarray(loadings)
                if values.ndim == 2:
                    table = pd.DataFrame(values, index=names, columns=[f'PC{i + 1}' for i in range(values.shape[1])])
                    with st.expander('Top five signed loadings per component'):
                        for pc in table:
                            top = table[pc].abs().nlargest(5).index
                            st.write(pc + ': ' + ', '.join(f'{name} ({table.loc[name, pc]:+.3f})' for name in top))
with tab5:
    st.header('Monthly repeat-purchase retention')
    st.caption('Full transaction base; global segment filters do not change this view. Cohort = first observed purchase month. Missing cells are unobserved; 0% means an observed period with no returning customers.')
    confirmed_complete = st.checkbox('I can confirm a data-complete-through date', value=False, key='retention_confirmed')
    complete_through = None
    if confirmed_complete:
        complete_through = st.date_input('Data complete through (source-confirmed)',
            value=finalized_month_end(max_date), max_value=month_end(max_date),
            key='retention_complete_date_' + st.session_state.tx_fingerprint)
    cutoff = finalized_month_end(max_date, complete_through)
    st.caption(f'Finalized through {cutoff.isoformat()}. Later periods and cohorts are excluded. '
               'Default excludes the latest observed month; this assumes earlier source months are complete.')
    if st.button('Generate retention matrix', type='primary'):
        st.session_state.show_retention = True
    if st.session_state.get('show_retention', False):
        with st.spinner('Computing cohort retention...'):
            retention_key = (st.session_state.tx_fingerprint, cutoff.isoformat())
            if (st.session_state.get('retention_key') != retention_key or st.session_state.get('retention_result') is None
                    or st.session_state.get('retention_cohort_sizes') is None):
                st.session_state.retention_result = retention_matrix(tx_lazy, cutoff)
                st.session_state.retention_cohort_sizes = cohort_sizes(tx_lazy)
                st.session_state.retention_key = retention_key
            retention = st.session_state.retention_result
            sizes = st.session_state.retention_cohort_sizes
        if retention.height > 0 and retention.width > 1:
            matrix = retention.to_pandas().set_index('Cohort')
            values = matrix.to_numpy(dtype=float)
            size_values = np.array([sizes.get(cohort, np.nan) for cohort in matrix.index], dtype=float)
            retained = np.rint(values * size_values[:, None])
            customdata = np.stack([retained, np.broadcast_to(size_values[:, None], values.shape)], axis=-1)
            row_labels = [f'{cohort} (n={int(n):,})' if np.isfinite(n) else str(cohort) for cohort, n in zip(matrix.index, size_values)]
            fig = px.imshow(values, x=[str(c) for c in matrix.columns], y=row_labels, text_auto='.0%', aspect='auto', color_continuous_scale='YlGnBu', range_color=[0, 1])
            fig.update_traces(customdata=customdata, hoverongaps=False,
                hovertemplate='Cohort: %{y}<br>Months since first observed purchase: %{x}<br>Retention: %{z:.1%}<br>Retained customers: %{customdata[0]:,.0f} of %{customdata[1]:,.0f}<extra></extra>')
            fig.update_yaxes(type='category')
            fig.update_layout(height=600, xaxis_title='Months since first observed purchase', yaxis_title='First observed purchase cohort')
            st.plotly_chart(fig, use_container_width=True)
            columns = st.columns(3)
            for column, period in zip(columns[:2], ['1', '3']):
                if period in retention.columns:
                    value = retention[period].drop_nulls().mean()
                    column.metric(f'Avg month {period} retention (finalized)', f'{value:.1%}' if value is not None else 'N/A')
            columns[2].metric('Cohorts', retention.height)
            st.caption('Row labels show each cohort starting customer count (distinct customer IDs, full base, independent of segment filters). Hover shows retained customers; blank cells are unfinished periods, not zero.')
            st.caption('Averages weight eligible finalized cohorts equally. Nulls are unfinished periods; observed zero-return periods remain 0%.')
        else:
            st.info('Insufficient transaction history for cohort analysis.')

st.sidebar.divider()
st.sidebar.header('Selected segment summaries')
for cluster in profiles.index:
    if cluster in selected:
        row = profiles.loc[cluster]
        st.sidebar.caption(f"{row['label']} · {int(row['customers']):,} customers · Median observed spend ${row['median_spend']:,.0f}")
