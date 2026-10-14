import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import io
import hashlib
import logging
from datetime import date, timedelta
from segmentation import AdvancedCustomerSegmentation
from data_safety import source_fingerprint, generate_session_segments, finalized_month_end, month_end, retention_matrix
from frontend_storytelling import (
    METRICS, prepare_customers, segment_profiles, render_contribution,
    render_profiles, render_risk, render_zscore_small_multiples,
)
from purchase_inactivity import (
    FEATURE_COLUMNS as INACTIVITY_FEATURE_COLUMNS,
    FEATURE_LABELS as INACTIVITY_FEATURE_LABELS,
    explain_customer,
    fit_evaluate_and_score,
    global_feature_effects,
    prepare_transactions as prepare_inactivity_transactions,
)
from customer_segment_DNA import DNAConfig, render_segment_dna, feature_catalog
from segment_migration import render_segment_migration

logger = logging.getLogger(__name__)

# ---- Plotly brand template ----
BRAND_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0F172A"),
        title=dict(font=dict(size=16, color="#0F172A"), x=0.02, xanchor="left"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=["#2563EB", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"],
        xaxis=dict(
            gridcolor="#E2E8F0", zerolinecolor="#E2E8F0", showgrid=True, zeroline=True,
            title=dict(font=dict(size=12, color="#0F172A")),
            tickfont=dict(size=11, color="#0F172A")
        ),
        yaxis=dict(
            gridcolor="#E2E8F0", zerolinecolor="#E2E8F0", showgrid=True, zeroline=True,
            title=dict(font=dict(size=12, color="#0F172A")),
            tickfont=dict(size=11, color="#0F172A"), automargin=True
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1,
            font=dict(size=11, color="#0F172A"), yanchor="top", y=1.0, xanchor="left", x=1.02
        ),
        margin=dict(l=60, r=120, t=50, b=40),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", font=dict(size=11, color="#0F172A")),
    )
)
pio.templates["brand"] = BRAND_TEMPLATE
pio.templates.default = "brand"

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation")
st.caption(
    "Explore customer value, purchasing behavior, retention, "
    "and estimated purchase inactivity."
)

for key, default in {
    "segments_df": None, "tx_df": None, "segments_generated": False,
    "tx_fingerprint": None, "segments_fingerprint": None, "pca_metadata": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def pca_variance_diagnostics(metadata, fingerprint):
    '''Validated (ratios, cumulative-or-None) arrays, or None if metadata is unusable.

    Requires a matching source fingerprint, finite ratios in [0, 1] summing to at most 1,
    and, when present, a cumulative array of the same length consistent with the ratios.'''
    if not isinstance(metadata, dict) or metadata.get('source_fingerprint') != fingerprint:
        return None
    try:
        ratios = np.asarray(metadata.get('explained_variance_ratio'), dtype=float)
        raw_cumulative = metadata.get('explained_variance_cumulative')
        cumulative = None if raw_cumulative is None else np.asarray(raw_cumulative, dtype=float)
    except (TypeError, ValueError):
        return None
    if (ratios.ndim != 1 or ratios.size == 0 or not np.isfinite(ratios).all()
            or (ratios < 0).any() or (ratios > 1).any() or ratios.sum() > 1 + 1e-6):
        return None
    if cumulative is not None:
        if (cumulative.shape != ratios.shape or not np.isfinite(cumulative).all()
                or not np.allclose(cumulative, np.cumsum(ratios), atol=1e-6)):
            return None
    return ratios, cumulative


def pca_variance_by_component(metadata, fingerprint):
    '''Explained-variance ratios by PC name, from the same validated diagnostics as the chart.'''
    diagnostics = pca_variance_diagnostics(metadata, fingerprint)
    if diagnostics is None:
        return {}
    return {f'PC{i + 1}': float(r) for i, r in enumerate(diagnostics[0])}


def pc_axis_label(pc, variance):
    return f'{pc} ({variance[pc]:.1%} of variance)' if pc in variance else pc


# ---- Risk palette ----
RISK_COLORS = {
    "Low (<20%)": "#287D70",
    "Medium (20-50%)": "#D97706",
    "High (50-80%)": "#C2413B",
    "Very High (>80%)": "#991B1B",
}


# ---- Required transaction columns for validation ----
REQUIRED_TX_COLUMNS = {
    "customer_id", "transaction_id", "product_id",
    "product_description", "department", "category",
    "transaction_date", "price", "quantity",
}

# Optional columns that can be mapped if present (not already required)
OPTIONAL_TX_COLUMNS = {
    # These are already in REQUIRED_TX_COLUMNS, so they're handled above
}

# Default column name suggestions for common variations
COLUMN_ALIASES = {
    "customer_id": ["customer_id", "customerid", "cust_id", "client_id", "user_id", "buyer_id", "customer id"],
    "transaction_id": ["transaction_id", "transactionid", "txn_id", "order_id", "orderid", "invoice_id", "invoice"],
    "product_id": ["product_id", "productid", "sku", "item_id", "itemid", "stockcode", "stock_code"],
    "product_description": ["product_description", "productdescription", "description", "product_name", "productname", "item_description"],
    "department": ["department", "dept", "division", "category_group"],
    "category": ["category", "cat", "product_category", "subcategory"],
    "transaction_date": ["transaction_date", "transactiondate", "date", "order_date", "orderdate", "purchase_date", "purchasedate", "txn_date", "transaction_datetime", "invoicedate", "invoice_date"],
    "price": ["price", "unit_price", "unitprice", "amount", "cost", "sale_price"],
    "quantity": ["quantity", "qty", "units", "quantity_sold", "units_sold"],
}


def validate_transactions(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Validate and clean uploaded transaction Parquet.
    
    Returns cleaned DataFrame and count of dropped rows.
    """
    missing = REQUIRED_TX_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )
    if df.is_empty():
        raise ValueError("The uploaded dataset contains no rows.")

    # Handle transaction_date conversion - may already be datetime or string
    if df["transaction_date"].dtype not in (pl.Datetime, pl.Date):
        df = df.with_columns(pl.col("transaction_date").str.to_datetime(strict=False))

    df = df.with_columns([
        pl.col("customer_id").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("transaction_id").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("product_id").cast(pl.String, strict=False).str.strip_chars(),
        pl.col("product_description")
            .cast(pl.String, strict=False).fill_null(""),
        pl.col("department")
            .cast(pl.String, strict=False).fill_null("__UNKNOWN__"),
        pl.col("category")
            .cast(pl.String, strict=False).fill_null("__UNKNOWN__"),
        pl.col("price").cast(pl.Float64, strict=False),
        pl.col("quantity").cast(pl.Float64, strict=False),
    ])

    # Ensure string ID columns are actually strings (catch bad column mappings)
    for col in ["customer_id", "transaction_id", "product_id"]:
        if df[col].dtype != pl.String:
            try:
                # If it's datetime, convert to ISO format string
                if df[col].dtype in (pl.Datetime, pl.Date):
                    df = df.with_columns(pl.col(col).dt.strftime("%Y-%m-%d"))
                else:
                    df = df.with_columns(pl.col(col).cast(pl.String))
            except Exception:
                raise ValueError(
                    f"Column '{col}' could not be converted to string. "
                    f"Please check your column mapping - the selected column "
                    f"may contain incompatible data (e.g., dates instead of IDs)."
                )

    valid = (
        pl.col("customer_id").is_not_null() & (pl.col("customer_id") != "") &
        pl.col("transaction_id").is_not_null() & (pl.col("transaction_id") != "") &
        pl.col("product_id").is_not_null() & (pl.col("product_id") != "") &
        pl.col("transaction_date").is_not_null() &
        pl.col("price").is_finite() & (pl.col("price") >= 0) &
        pl.col("quantity").is_finite() & (pl.col("quantity") > 0)
    )
    cleaned = df.filter(valid)
    dropped_rows = df.height - cleaned.height
    if cleaned.is_empty():
        raise ValueError(
            "No valid transaction rows remain after validation."
        )
    return cleaned, dropped_rows


def reset_inactivity_state():
    """Clear all inactivity-model session state when the source dataset changes."""
    keys = (
        "inactivity_prepared_tx",
        "inactivity_prepared_tx_fingerprint",
        "inactivity_file_fingerprint",
        "inactivity_result",
        "inactivity_run_key",
        "inactivity_error",
        "inactivity_error_key",
        "inactivity_coverage_confirmed",
        "inactivity_coverage_date",
    )
    for key in keys:
        st.session_state.pop(key, None)


def suggest_column_mapping(df_columns: list[str]) -> dict[str, str]:
    """Suggest mappings from dataframe columns to required columns based on aliases."""
    mapping = {}
    df_cols_lower = {c.lower(): c for c in df_columns}
    
    for required_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in df_cols_lower:
                mapping[required_col] = df_cols_lower[alias.lower()]
                break
    return mapping


def apply_column_mapping(df: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    """Apply column mapping by renaming columns in the DataFrame."""
    # Create reverse mapping: source_col -> target_col
    rename_dict = {v: k for k, v in mapping.items() if v in df.columns}
    if rename_dict:
        df = df.rename(rename_dict)
    return df


def cohort_sizes(lazy_transactions):
    '''Distinct customers per first-purchase month, from the full transaction base.'''
    sizes = (lazy_transactions.group_by('customer_id')
        .agg(pl.col('transaction_date').dt.truncate('1mo').min().alias('CohortMonth'))
        .filter(pl.col('CohortMonth').is_not_null())
        .group_by('CohortMonth').agg(pl.col('customer_id').n_unique().alias('size')).collect())
    return {month.strftime('%Y-%m'): int(n) for month, n in sizes.iter_rows() if month is not None}


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
    reset_inactivity_state()

st.sidebar.header("Data upload")
uploaded_transactions = st.sidebar.file_uploader(
    "Cleaned transactions (Parquet)", type=["parquet"],
    help="Uploads and generated results stay in this session; the app does not save upload artifacts to disk.",
)

# Expected input format help
with st.sidebar.expander("Expected input format"):
    st.write(
        "Upload a cleaned Parquet file containing one row "
        "per transaction line."
    )
    st.code(
        "customer_id, transaction_id, product_id, "
        "product_description, department, category, "
        "transaction_date, price, quantity"
    )
    st.caption(
        "Dates must be valid; price must be non-negative; "
        "quantity must be positive."
    )

if uploaded_transactions is not None:
    new_fp = source_fingerprint(uploaded_transactions)
    if st.session_state.tx_fingerprint != new_fp:
        # Read the new dataset
        try:
            raw_df = pl.read_parquet(
                io.BytesIO(uploaded_transactions.getvalue())
            )
        except Exception:
            st.sidebar.error(
                "The upload could not be processed. Check that it is a "
                "readable Parquet file."
            )
            st.stop()

        # Column mapping UI
        st.sidebar.subheader("Column Mapping")
        st.sidebar.caption("Map your columns to the required schema. Auto-detected mappings are pre-filled.")
        
        detected_cols = raw_df.columns
        suggested_mapping = suggest_column_mapping(detected_cols)
        
        # Build the mapping UI
        final_mapping = {}
        cols_to_map = list(REQUIRED_TX_COLUMNS)
        
        # Show required columns first
        for req_col in sorted(cols_to_map):
            default = suggested_mapping.get(req_col, "")
            options = [""] + detected_cols
            try:
                default_idx = options.index(default) if default else 0
            except ValueError:
                default_idx = 0
            
            selected = st.sidebar.selectbox(
                f"{req_col} (required)",
                options=options,
                index=default_idx,
                key=f"map_{req_col}_{new_fp[:8]}",
                help=f"Select the column in your data that corresponds to '{req_col}'"
            )
            if selected:
                final_mapping[req_col] = selected
        
        # Apply mapping and validate
        mapped_df = apply_column_mapping(raw_df, final_mapping)
        
        try:
            new_tx_df, dropped_rows = validate_transactions(mapped_df)
        except Exception as e:
            st.sidebar.error(f"Validation failed: {e}")
            st.stop()
        
        if dropped_rows:
            st.sidebar.warning(
                f"{dropped_rows:,} invalid transaction rows were excluded."
            )
        
        reset_inactivity_state()
        st.session_state.tx_df = new_tx_df
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
    # Demo dataset selector - only includes files that exist in the repo
    demo_files = {
        "Online Retail II (cleaned)": "sample_data/online_retail_ii_cleaned.parquet",
    }
    selected_demo = st.sidebar.selectbox(
        "Demo dataset",
        options=list(demo_files.keys()),
        index=0,
        help="Select a built-in demo dataset to explore without uploading.",
    )
    demo_path = demo_files[selected_demo]
    
    import os
    if not os.path.exists(demo_path):
        st.sidebar.error(f"Demo file not found: {demo_path}")
        st.sidebar.caption(f"Working dir: {os.getcwd()}")
        st.sidebar.caption(f"Files in sample_data/: {os.listdir('sample_data') if os.path.exists('sample_data') else 'N/A'}")
        st.info("Upload cleaned transactions from the sidebar to begin.")
        st.stop()
    
    try:
        raw_df = pl.read_parquet(demo_path)
        
        # Apply column mapping for demo files (same as upload)
        suggested_mapping = suggest_column_mapping(raw_df.columns)
        mapped_df = apply_column_mapping(raw_df, suggested_mapping)
        
        new_tx_df, dropped_rows = validate_transactions(mapped_df)
        st.session_state.tx_df = new_tx_df
        st.session_state.tx_fingerprint = source_fingerprint(demo_path)
        if dropped_rows:
            st.sidebar.warning(
                f"{dropped_rows:,} invalid transaction rows were excluded from the demo file."
            )
    except Exception as e:
        st.sidebar.error(f"Demo file error: {type(e).__name__}: {e}")
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

# Reuse prepared presentation data within the session
presentation_fp = st.session_state.tx_fingerprint
if st.session_state.get("presentation_fp") != presentation_fp:
    try:
        customers = prepare_customers(segments_df.to_pandas())
    except ValueError as error:
        st.error(str(error))
        st.stop()
    profiles = segment_profiles(customers)
    st.session_state["presentation_customers"] = customers
    st.session_state["presentation_profiles"] = profiles
    st.session_state["presentation_fp"] = presentation_fp
customers = st.session_state["presentation_customers"]
profiles = st.session_state["presentation_profiles"]
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

tab1, tab2, tab3, tab4, tab5, tab6, tab_dna = st.tabs(
    [
        "Risk–Value Portfolio",
        "Segment Profiles",
        "Distributions",
        "Model Exploration",
        "Retention",
        "Purchase Inactivity",
        "Segment DNA",
    ]
)

with tab_dna:
    dna_currency = st.text_input(
        "Currency code / monetary unit",
        value="CU",
        key="segment_dna_currency",
        help=(
            "Enter the currency used by your source data, such as GBP, "
            "EUR, or HUF. CU means unspecified currency units. "
            "This labels values; it does not convert currencies."
        ),
    ).strip() or "CU"

    show_purchase_details = st.checkbox(
        "Include category mix and monthly spend trends",
        value=False,
        key="segment_dna_purchase_details",
        help="Enable this to include transaction-level purchase charts.",
    )

    dna_transactions = None

    if show_purchase_details:
        # Debug diagnostics in an expander
        with st.expander("Transaction data diagnostics", expanded=False):
            st.caption(f"Available columns in tx_df: {tx_df.columns}")
            try:
                dna_transactions = tx_df.select(
                    [
                        "customer_id",
                        "transaction_id",
                        "transaction_date",
                        "price",
                        "quantity",
                        "category",
                    ]
                ).to_pandas()
                st.caption(f"Transaction rows: {len(dna_transactions)}")
            except Exception as e:
                st.error(f"Failed to select transaction columns: {e}")
                dna_transactions = None

    # Reuse Polars-to-Pandas conversion within the user session
    dna_cfg = DNAConfig(currency=dna_currency)

    dna_cols = list(dict.fromkeys([
        dna_cfg.customer_col,
        dna_cfg.segment_col,
        *[
            c for c in feature_catalog(dna_cfg)
            if c in segments_df.columns
        ],
    ]))

    cached = st.session_state.get("_dna_customer_input")

    if (
        cached is None
        or cached["source"] is not segments_df
        or cached["columns"] != dna_cols
    ):
        with st.spinner("Preparing Segment DNA input..."):
            cached = {
                "source": segments_df,
                "columns": dna_cols,
                "frame": segments_df.select(dna_cols).to_pandas(),
            }
            st.session_state["_dna_customer_input"] = cached

    render_segment_dna(
        customers=cached["frame"],
        transactions=dna_transactions,
        config=dna_cfg,
        eligible_segments=selected,
        segment_labels=labels,
        key_prefix=f"segment_dna_{st.session_state.tx_fingerprint}",
    )

    st.divider()

    render_segment_migration(
        tx_df,
        source_fingerprint=st.session_state.tx_fingerprint,
        key_prefix="segment_dna_migration",
    )

with tab1:
    render_risk(customers, selected, colors, order, st.session_state.tx_fingerprint, currency="$")
with tab2:
    render_profiles(customers, profiles, selected)
    render_zscore_small_multiples(customers, profiles, selected, colors, order)
with tab3:
    st.header('Distributions within segments')
    st.caption('Segment averages hide variation. Compare distributions before interpreting a persona.')
    metrics = [m for m in METRICS if m in view.columns]
    metric = st.selectbox('Metric', metrics, format_func=lambda m: METRICS[m], key='distribution_metric')
    plot_type = st.radio('Plot type', ['Boxplot', 'Violin plot'], index=0, horizontal=True, key='distribution_plot_type')
    log = st.checkbox('Logarithmic axis', value=metric in ['Total_Lifetime_Spend', 'Average_Order_Value_AOV', 'Total_Lifetime_Transactions'], key=f'distribution_log_{metric}')
    chart_name = 'violin plot' if plot_type == 'Violin plot' else 'boxplot'
    stats_view = view.assign(**{metric: view[metric].replace([np.inf, -np.inf], np.nan)})
    finite = stats_view[stats_view[metric].notna()]
    dist = finite[finite[metric].gt(0)] if log else finite
    if len(finite) < len(view):
        st.caption(f'{len(view) - len(finite):,} missing or non-finite values are not drawn; summary statistics also exclude them.')
    if log and len(dist) < len(finite):
        st.caption(f'{len(finite) - len(dist):,} non-positive observations omitted from the logarithmic {chart_name}; summary statistics include them.')
    if dist.empty:
        st.info('No plottable observations for this metric and axis setting.')
    else:
        plot_options = dict(x='Segment', y=metric, color='Segment', color_discrete_map=colors, category_orders=category_orders, points='outliers', log_y=log, labels={metric: METRICS[metric]})
        fig = px.violin(dist, box=True, **plot_options) if plot_type == 'Violin plot' else px.box(dist, **plot_options)
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    stats = stats_view.groupby('Segment')[metric].agg(['median', 'mean', 'std', 'count']).reindex(order)
    st.dataframe(stats.rename(columns={'median': 'Median', 'mean': 'Mean', 'std': 'Std dev', 'count': 'Customers'}), use_container_width=True)
with tab4:
    st.header('Model exploration: PCA')
    st.caption('Projections show only part of the clustering space; visual overlap does not establish full-space overlap.')
    pca_metadata = st.session_state.pca_metadata
    diagnostics = pca_variance_diagnostics(pca_metadata, st.session_state.tx_fingerprint)
    variance = pca_variance_by_component(pca_metadata, st.session_state.tx_fingerprint)
    if {'PC1', 'PC2', 'PC3'}.issubset(view.columns):
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
        st.plotly_chart(fig, use_container_width=True, theme=None)
        missing_axes = [pc for pc in axes if pc not in variance]
        if not variance:
            st.caption('Axis variance percentages are unavailable because PCA metadata is missing, invalid, or does not match these transactions. Diamonds labeled C# mark segment centers (mean PC coordinates).')
        elif missing_axes:
            st.caption(f"Variance percentage is unavailable for {', '.join(missing_axes)} in the validated PCA metadata; other axes show explained-variance ratios. Diamonds labeled C# mark segment centers (mean PC coordinates).")
        else:
            st.caption('Axis percentages are explained-variance ratios from validated, fingerprint-matched PCA metadata. Diamonds labeled C# mark segment centers: mean PC coordinates of the displayed customers in each segment.')
    else:
        st.info('PCA projections require PC1, PC2, and PC3 in the generated segments.')
    if st.checkbox(
            'Show explained variance and loadings',
            value=False,
            key='show_pca_diagnostics',
        ):
        metadata = pca_metadata
        if not metadata:
            st.info('Generate segments to populate PCA metadata.')
        elif metadata.get('source_fingerprint') != st.session_state.tx_fingerprint:
            st.warning('PCA metadata does not match these transactions. Refresh segments.')
        else:
            if diagnostics is None:
                st.warning('Explained-variance data is missing, malformed, or inconsistent, so the variance chart is unavailable.')
            else:
                evr, evc = diagnostics
                pcs = [f'PC{i + 1}' for i in range(len(evr))]
                fig = go.Figure(go.Bar(x=pcs, y=evr * 100, name='Individual'))
                if evc is not None:
                    fig.add_trace(go.Scatter(x=pcs, y=evc * 100, name='Cumulative', mode='lines+markers', yaxis='y2'))
                    st.caption(f'Total explained variance ({len(evr)} components): {evc[-1]:.1%}')
                fig.update_layout(height=350, yaxis_title='Individual variance (%)', yaxis2=dict(title='Cumulative (%)', overlaying='y', side='right', range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True, theme=None)
            loadings, names = metadata.get('loadings'), metadata.get('feature_names')
            try:
                values = np.asarray(loadings, dtype=float) if loadings is not None else None
            except (TypeError, ValueError):
                values = None
            if (values is not None and names and values.ndim == 2 and values.shape[0] == len(names)
                    and np.isfinite(values).all()):
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
            st.plotly_chart(fig, use_container_width=True, theme=None)
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

with tab6:
    st.header('30-Day Purchase Inactivity Model')
    st.caption(
        'A lightweight, explainable baseline that scores customers who purchased in the last 30 days '
        'for the likelihood of making no purchase in the following 30 days.'
    )
    st.info(
        'This is a probabilistic ranking, not a guarantee. The positive class is **no purchase in the next 30 days**. '
        'A predicted inactivity score is not a diagnosis of churn, and explanations describe model associations—not causes.'
    )

    # Prepare transactions for inactivity model (invalidate when source fingerprint changes)
    current_fp = st.session_state.tx_fingerprint
    if (st.session_state.get('inactivity_prepared_tx_fingerprint') != current_fp
            or 'inactivity_prepared_tx' not in st.session_state):
        with st.spinner('Preparing transactions for inactivity model…'):
            st.session_state['inactivity_prepared_tx'] = prepare_inactivity_transactions(tx_df)
            st.session_state['inactivity_prepared_tx_fingerprint'] = current_fp
    prepared_tx = st.session_state['inactivity_prepared_tx']

    latest_transaction_date = prepared_tx['order_date'].max().date()
    st.caption(
        f'Valid transaction lines: {len(prepared_tx):,} | Customers: {prepared_tx["customer_id"].nunique():,} | '
        f'Latest observed transaction date: {latest_transaction_date.isoformat()}'
    )

    st.subheader('Data coverage')
    st.warning(
        'The latest transaction date does not prove that the export is complete through that day. '
        'Confirm the source extract\'s actual coverage date before generating labels or scores.'
    )
    coverage_confirmed = st.checkbox(
        'I confirm the export is complete through the date below.', value=False, key='inactivity_coverage_confirmed'
    )
    today = date.today()
    if latest_transaction_date > today:
        st.error(
            "The dataset contains transaction dates later than "
            "the current date. Verify the source data before scoring."
        )
        st.stop()
    coverage_date = st.date_input(
        'Data complete through',
        value=latest_transaction_date,
        min_value=latest_transaction_date,
        max_value=today,
        help=(
            'Use the source system\'s confirmed completeness date, '
            'not merely the date of the last purchase.'
        ),
        key='inactivity_coverage_date',
    )

    if not coverage_confirmed:
        st.warning('Confirm data completeness above to enable training and scoring.')
        st.stop()

    # Derive fingerprint directly from active dataset (no stale stored fingerprint)
    file_fingerprint = st.session_state.tx_fingerprint
    run_key = (file_fingerprint, coverage_date.isoformat())

    if st.session_state.get('inactivity_error_key') != run_key:
        st.session_state.pop('inactivity_error', None)
        st.session_state.pop('inactivity_error_key', None)
    if st.button('Train model and score active customers', type='primary', use_container_width=True, key='inactivity_train_button'):
        with st.spinner('Building historical snapshots, testing the model, and scoring active customers…'):
            try:
                result = fit_evaluate_and_score(tx_df, coverage_date)
                st.session_state['inactivity_result'] = result
                st.session_state['inactivity_run_key'] = run_key
                st.session_state.pop('inactivity_error', None)
                st.session_state.pop('inactivity_error_key', None)
            except Exception:
                logger.exception(
                    "Inactivity model failed for source %s",
                    st.session_state.tx_fingerprint,
                )
                st.session_state['inactivity_error'] = (
                    "The model could not be trained. Check the data coverage "
                    "and required columns, then try again."
                )
                st.session_state['inactivity_error_key'] = run_key
                st.session_state.pop('inactivity_result', None)
                st.session_state.pop('inactivity_run_key', None)

    if (st.session_state.get('inactivity_error')
            and st.session_state.get('inactivity_error_key') == run_key):
        st.error(st.session_state['inactivity_error'])
        st.stop()

    result = st.session_state.get('inactivity_result')
    if result is None or st.session_state.get('inactivity_run_key') != run_key:
        st.info('Choose a confirmed completeness date, then click **Train model and score active customers**.')
        st.stop()

    evaluation = result['evaluation']
    st.subheader('Out-of-time validation')
    st.caption(
        'The latest historical snapshot windows are held out for evaluation; the model is not randomly splitting '
        'rows across time. This better approximates scoring customers after an earlier training period.'
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric('Historical snapshots', f"{evaluation['n_snapshots']:,}")
    metric_cols[1].metric('Holdout rows', f"{evaluation['n_test']:,}")
    metric_cols[2].metric(
        'ROC-AUC',
        f"{evaluation['roc_auc']:.3f}" if evaluation['roc_auc'] is not None else 'N/A',
        help='Measures ranking separation between the two classes; N/A when the holdout contains only one class.',
    )
    metric_cols[3].metric(
        'Average precision',
        f"{evaluation['average_precision']:.3f}" if evaluation['average_precision'] is not None else 'N/A',
        help='Precision-recall summary for the no-purchase class; N/A when the holdout contains only one class.',
    )
    st.caption(
        f"Holdout snapshot dates: {evaluation['test_start']} through {evaluation['test_end']} | "
        f"No-purchase share in holdout: {evaluation['test_no_purchase_rate']:.1%} | "
        f"Brier score: {evaluation['brier_score']:.3f} (lower is better; probability calibration has not been independently validated)."
    )
    if evaluation['n_snapshots'] < 6:
        st.warning('There are relatively few historical snapshots. Treat holdout metrics as preliminary; more complete transaction history will make evaluation more reliable.')
    if not evaluation['holdout_has_both_classes']:
        st.warning('The time-based holdout contains only one target class, so ROC-AUC and average precision cannot be reported reliably for this split.')

    # Model evaluation curves
    if evaluation['holdout_has_both_classes'] and 'holdout_y_true' in result and 'holdout_y_prob' in result:
        y_true = result['holdout_y_true']
        y_prob = result['holdout_y_prob']
        
        st.subheader('Model evaluation curves')
        from sklearn.metrics import roc_curve, precision_recall_curve
        from sklearn.calibration import calibration_curve
        
        curve_col1, curve_col2, curve_col3 = st.columns(3)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={evaluation["roc_auc"]:.3f})', line=dict(color='#2563EB', width=2)))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='#94A3B8', width=1, dash='dash')))
        roc_fig.update_layout(
            title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            height=300, legend=dict(yanchor='bottom', y=0.01, xanchor='right', x=0.99),
            margin=dict(l=40, r=20, t=40, b=40)
        )
        curve_col1.plotly_chart(roc_fig, use_container_width=True, theme=None)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AP={evaluation["average_precision"]:.3f})', line=dict(color='#EF553B', width=2)))
        # Baseline (prevalence)
        baseline = y_true.mean()
        pr_fig.add_hline(y=baseline, line_dash='dash', line_color='#94A3B8', annotation_text=f'Baseline ({baseline:.1%})', annotation_position='bottom right')
        pr_fig.update_layout(
            title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision',
            height=300, legend=dict(yanchor='bottom', y=0.01, xanchor='right', x=0.99),
            margin=dict(l=40, r=20, t=40, b=40), yaxis_range=[0, 1.05]
        )
        curve_col2.plotly_chart(pr_fig, use_container_width=True, theme=None)
        
        # Calibration (Reliability) Plot
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Model', line=dict(color='#00CC96', width=2), marker=dict(size=6)))
        cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfectly calibrated', line=dict(color='#94A3B8', width=1, dash='dash')))
        cal_fig.update_layout(
            title='Calibration Plot', xaxis_title='Mean predicted probability', yaxis_title='Fraction of positives',
            height=300, legend=dict(yanchor='bottom', y=0.01, xanchor='right', x=0.99),
            margin=dict(l=40, r=20, t=40, b=40)
        )
        curve_col3.plotly_chart(cal_fig, use_container_width=True, theme=None)
        
        st.caption('Calibration shows whether predicted probabilities match observed frequencies. A well-calibrated model falls on the diagonal. Brier score alone does not establish calibration.')

    predictions = result['predictions']
    current_features = result['current_features']
    st.subheader('Currently active customers ranked by inactivity score')
    st.caption(
        f"Scored {len(predictions):,} customers with at least one purchase in the 30 days ending "
        f"{coverage_date.isoformat()}. Scores are model-estimated probabilities, not certainties."
    )
    if predictions.empty:
        st.warning('No customers had a purchase within the last 30 days of the confirmed coverage date.')
        st.stop()

    risk_col, active_col = st.columns(2)
    with risk_col:
        st.metric('Customers scored', f'{len(predictions):,}')
    with active_col:
        st.metric('Mean estimated no-purchase probability', f'{predictions["no_purchase_probability"].mean():.1%}')

    # Score histogram with risk bands
    predictions = predictions.copy()
    predictions['Risk band'] = pd.cut(
        predictions['no_purchase_probability'],
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=['Low (<20%)', 'Medium (20-50%)', 'High (50-80%)', 'Very High (>80%)'],
        include_lowest=True
    )
    hist = px.histogram(
        predictions,
        x='no_purchase_probability',
        color='Risk band',
        nbins=20,
        color_discrete_map=RISK_COLORS,
        labels={
            'no_purchase_probability': 'Estimated probability of no purchase in the next 30 days',
            'count': 'Customers',
        },
        title='Distribution of inactivity scores by risk band',
    )
    hist.update_layout(height=330, yaxis_title='Customers', bargap=0.05)
    st.plotly_chart(hist, use_container_width=True, theme=None)

    shown = predictions.head(100).copy()
    shown['no_purchase_probability'] = shown['no_purchase_probability'].map(lambda x: f'{x:.1%}')
    feature_lookup = current_features.set_index('customer_id')
    for col in ['orders_30d', 'spend_30d', 'orders_90d', 'spend_90d', 'days_since_last_purchase']:
        if col in feature_lookup:
            shown[col] = shown['customer_id'].map(feature_lookup[col])
    shown = shown.rename(columns={
        'customer_id': 'Customer ID',
        'no_purchase_probability': 'Estimated no-purchase probability',
        'orders_30d': 'Orders (30d)',
        'spend_30d': 'Spend (30d)',
        'orders_90d': 'Orders (90d)',
        'spend_90d': 'Spend (90d)',
        'days_since_last_purchase': 'Days since last purchase',
    })
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button(
        'Download all scored customers (CSV)',
        data=predictions.to_csv(index=False).encode('utf-8'),
        file_name='purchase_inactivity_scores.csv',
        mime='text/csv',
    )

    st.subheader('Why did the model give this score?')
    customer_ids = predictions['customer_id'].astype(str).tolist()
    selected_customer = st.selectbox('Choose a customer to explain', customer_ids, key='inactivity_customer_select')
    local = explain_customer(result['model'], current_features, selected_customer, top_n=len(INACTIVITY_FEATURE_COLUMNS))
    local_plot = local.sort_values('contribution_log_odds', ascending=True).copy()
    local_plot['effect'] = np.where(
        local_plot['contribution_log_odds'].ge(0),
        'Pushes toward no purchase',
        'Pushes away from no purchase',
    )
    fig = px.bar(
        local_plot,
        x='contribution_log_odds',
        y='metric',
        color='effect',
        orientation='h',
        color_discrete_map={
            'Pushes toward no purchase': '#C2413B',
            'Pushes away from no purchase': '#287D70',
        },
        labels={
            'contribution_log_odds': 'Contribution to model log-odds (relative to baseline)',
            'metric': 'Metric',
            'effect': 'Model effect',
        },
        title=f'Largest model contributions for customer {selected_customer}',
    )
    fig.add_vline(x=0, line_color='gray', line_width=1)
    fig.update_layout(height=max(420, len(local_plot) * 28), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    explanation_table = local[['metric', 'value', 'contribution_log_odds', 'effect']].copy()
    explanation_table['value'] = explanation_table.apply(
        lambda r: f'${r["value"]:,.2f}' if 'spend' in str(r['metric']).lower() or 'value' in str(r['metric']).lower() else f'{r["value"]:,.2f}',
        axis=1,
    )
    st.dataframe(explanation_table, use_container_width=True, hide_index=True)
    st.caption(
        'Positive contributions push this model\'s score toward \'no purchase\'; negative contributions push it away. '
        'Contributions are additive in log-odds after feature scaling. They explain this fitted Logistic Regression, '
        'not causation, and correlated metrics can share or offset contributions.'
    )

    st.subheader('Which metrics matter most overall?')
    global_effects = global_feature_effects(result['model'])
    top_global = global_effects.head(12).sort_values('coefficient_per_std')
    top_global['effect'] = np.where(
        top_global['coefficient_per_std'].ge(0),
        'Higher values associate with more inactivity',
        'Higher values associate with less inactivity',
    )
    global_fig = px.bar(
        top_global,
        x='coefficient_per_std',
        y='metric',
        color='effect',
        orientation='h',
        color_discrete_map={
            'Higher values associate with more inactivity': '#C2413B',
            'Higher values associate with less inactivity': '#287D70',
        },
        labels={'coefficient_per_std': 'Log-odds change per 1 SD increase', 'metric': 'Metric'},
        title='Largest standardized Logistic Regression coefficients',
    )
    global_fig.add_vline(x=0, line_color='gray', line_width=1)
    global_fig.update_layout(height=450)
    st.plotly_chart(global_fig, use_container_width=True, theme=None)
    st.dataframe(
        global_effects[['metric', 'coefficient_per_std', 'odds_ratio_per_std', 'direction']].head(15),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        'Global coefficients are conditional associations in this model. A positive coefficient means that a one-standard-deviation '
        'increase in that metric is associated with higher no-purchase odds, holding the other model inputs fixed. '
        'This is not a causal explanation or proof that changing the metric would change future behavior.'
    )

st.sidebar.divider()
st.sidebar.header('Selected segment summaries')
for cluster in profiles.index:
    if cluster in selected:
        row = profiles.loc[cluster]
        st.sidebar.caption(f"{row['label']} · {int(row['customers']):,} customers · Median observed spend ${row['median_spend']:,.0f}")
