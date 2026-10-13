import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import hashlib
import json
from datetime import timedelta
from segmentation import AdvancedCustomerSegmentation
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


def compute_file_fingerprint(uploaded_file):
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.sha256(content).hexdigest()[:16]


def compute_local_file_fingerprint(path):
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def run_segmentation_pipeline(transactions_df, tx_fingerprint):
    pipeline = AdvancedCustomerSegmentation(max_clusters=8, n_components=6)
    segments = pipeline.fit_predict(transactions_df)
    return segments, pipeline.get_pca_metadata(tx_fingerprint)


def get_or_generate_segments(tx_df, tx_fingerprint, force_refresh=False):
    if not force_refresh:
        try:
            segments = pl.read_parquet("customer_segments_advanced.parquet")
            with open("pca_metadata.json", "r") as file:
                metadata = json.load(file)
            if metadata.get("source_fingerprint") == tx_fingerprint:
                st.sidebar.success("Loaded matching local segments")
                return segments, metadata
            st.sidebar.warning("Local segments do not match. Regenerating...")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    if tx_df is None:
        st.error("Upload transaction data first.")
        st.stop()
    try:
        with st.sidebar.status("Running segmentation...", expanded=True) as status:
            status.write("Engineering behavioral metrics, robust preprocessing, PCA, and KMeans...")
            segments, metadata = run_segmentation_pipeline(tx_df, tx_fingerprint)
            segments.write_parquet("customer_segments_advanced.parquet")
            with open("pca_metadata.json", "w") as file:
                json.dump(metadata, file, indent=2)
            status.update(label="Segments generated and saved", state="complete")
        return segments, metadata
    except Exception as error:
        st.error(f"Segmentation failed: {error}")
        st.stop()


@st.cache_data
def generate_retention_matrix_polars(transactions_lf):
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
        values='customer_count', index='CohortMonth', on='PeriodNumber',
        aggregate_function='first'
    ).sort('CohortMonth')
    period_cols_int = []
    for column in cohort_pivot.columns:
        if column != 'CohortMonth':
            try:
                period_cols_int.append(int(column))
            except (ValueError, TypeError):
                pass
    if period_cols_int:
        period_cols_int.sort()
        max_order_month = df.select(pl.col('OrderMonth').max()).collect().item()
        if 0 in period_cols_int:
            for period in period_cols_int:
                if period != 0:
                    column = str(period)
                    cohort_pivot = cohort_pivot.with_columns(
                        pl.when(
                            (pl.col('0') > 0) &
                            ((pl.col('CohortMonth').dt.year() * 12 + pl.col('CohortMonth').dt.month() + period) <=
                             (max_order_month.year() * 12 + max_order_month.month()))
                        ).then(
                            pl.when(pl.col(column).is_not_null())
                            .then(pl.col(column) / pl.col('0')).otherwise(pl.lit(0.0))
                        ).otherwise(None).alias(column)
                    )
            cohort_pivot = cohort_pivot.with_columns(
                pl.when(pl.col('0') > 0).then(pl.lit(1.0)).otherwise(None).alias('0')
            )
    return cohort_pivot.with_columns(
        pl.col('CohortMonth').dt.strftime('%Y-%m').alias('Cohort')
    ).drop('CohortMonth').select(['Cohort'] + [str(p) for p in period_cols_int])


st.sidebar.header("Data upload")
uploaded_transactions = st.sidebar.file_uploader(
    "Cleaned transactions (Parquet)", type=["parquet"],
    help="Upload cleaned transaction data; segmentation runs only on request.",
)
if uploaded_transactions is not None:
    new_fp = compute_file_fingerprint(uploaded_transactions)
    if st.session_state.tx_fingerprint != new_fp:
        st.session_state.tx_df = pl.read_parquet(io.BytesIO(uploaded_transactions.read()))
        st.session_state.tx_fingerprint = new_fp
        st.session_state.segments_generated = False
        st.session_state.segments_df = None
        st.session_state.segments_fingerprint = None
        st.session_state.pca_metadata = None
        st.session_state.show_retention = False
        st.sidebar.info("New data detected. Generate segments below.")
elif st.session_state.tx_df is None:
    try:
        st.session_state.tx_df = pl.read_parquet("customer_transactions_clean.parquet")
        st.session_state.tx_fingerprint = compute_local_file_fingerprint("customer_transactions_clean.parquet")
    except FileNotFoundError:
        st.info("Upload cleaned transactions from the sidebar to begin.")
        st.stop()
tx_df = st.session_state.tx_df
tx_lazy = tx_df.lazy()

st.sidebar.header("Segmentation controls")
ready = st.session_state.segments_generated and st.session_state.segments_df is not None
refresh = st.sidebar.button("Refresh segments" if ready else "Generate segments", type="primary", use_container_width=True)
force = st.sidebar.button("Force regenerate", help="Bypass both saved segments and the cached pipeline result.", use_container_width=True)
if refresh or force:
    if force:
        run_segmentation_pipeline.clear()
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
    log = st.checkbox('Logarithmic axis', value=metric in ['Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Total_Lifetime_Transactions'], key=f'distribution_log_{metric}')
    dist = view if not log else view[view[metric].gt(0)]
    if log and len(dist) < len(view):
        st.caption(f'{len(view) - len(dist):,} non-positive values omitted from the log chart; summary statistics include them.')
    if not dist.empty:
        fig = px.box(dist, x='Segment', y=metric, color='Segment', color_discrete_map=colors, category_orders=category_orders, points='outliers', log_y=log, labels={metric: METRICS[metric]})
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    stats = view.groupby('Segment')[metric].agg(['median', 'mean', 'std', 'count']).reindex(order)
    st.dataframe(stats.rename(columns={'median': 'Median', 'mean': 'Mean', 'std': 'Std dev', 'count': 'Customers'}), use_container_width=True)
with tab4:
    st.header('Model exploration: PCA')
    st.caption('Projections show only part of the clustering space; visual overlap does not establish full-space overlap.')
    if {'PC1', 'PC2', 'PC3'}.issubset(view.columns):
        projection = st.radio('Projection', ['PC1 vs PC2', 'PC1 vs PC3', '3D (PC1/PC2/PC3)'], horizontal=True)
        options = dict(color='Segment', color_discrete_map=colors, category_orders=category_orders, hover_name='customer_id', hover_data=['Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Primary_Category_Name'], opacity=.65)
        fig = px.scatter_3d(view, x='PC1', y='PC2', z='PC3', **options) if projection.startswith('3D') else px.scatter(view, x='PC1', y='PC3' if projection.endswith('PC3') else 'PC2', **options)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('PCA projections require PC1, PC2, and PC3 in the generated segments.')
    if st.checkbox('Show explained variance and loadings', value=True):
        metadata = st.session_state.pca_metadata
        if metadata is None:
            try:
                with open('pca_metadata.json') as file:
                    metadata = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError):
                metadata = None
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
    if st.button('Generate retention matrix', type='primary'):
        st.session_state.show_retention = True
    if st.session_state.get('show_retention', False):
        with st.spinner('Computing cohort retention...'):
            retention = generate_retention_matrix_polars(tx_lazy)
        if retention.height > 0 and retention.width > 1:
            matrix = retention.to_pandas().set_index('Cohort')
            fig = px.imshow(matrix, text_auto='.0%', aspect='auto', color_continuous_scale='YlGnBu', range_color=[0, 1])
            fig.update_traces(hoverongaps=False)
            fig.update_layout(height=600, xaxis_title='Months since first observed purchase', yaxis_title='First observed purchase cohort')
            st.plotly_chart(fig, use_container_width=True)
            columns = st.columns(3)
            for column, period in zip(columns[:2], ['1', '3']):
                if period in retention.columns:
                    value = retention[period].drop_nulls().mean()
                    column.metric(f'Avg month {period} retention (observed)', f'{value:.1%}' if value is not None else 'N/A')
            columns[2].metric('Cohorts', retention.height)
            st.caption('Averages weight cohorts equally. The final calendar month may be incomplete; observed does not mean a fully completed month.')
        else:
            st.info('Insufficient transaction history for cohort analysis.')

st.sidebar.divider()
st.sidebar.header('Selected segment summaries')
for cluster in profiles.index:
    if cluster in selected:
        row = profiles.loc[cluster]
        st.sidebar.caption(f"{row['label']} · {int(row['customers']):,} customers · Median observed spend ${row['median_spend']:,.0f}")
