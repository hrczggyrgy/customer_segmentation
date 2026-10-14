"""Customer Segment DNA: standalone Streamlit app and embeddable panel.

Install:
    pip install "streamlit>=1.32" "pandas>=2.0" "numpy>=1.24" "plotly>=5.18" pyarrow
Run:
    streamlit run customer_segment_DNA.py

Embedded usage (call only after the host has checked source fingerprints):
    from customer_segment_DNA import DNAConfig, render_segment_dna
    render_segment_dna(
        segments_df.to_pandas(),
        transactions=tx_df.to_pandas(),
        config=DNAConfig(currency="GBP"),
        eligible_segments=selected,
        segment_labels=labels,
        key_prefix="segment_dna",
    )

Input contract
--------------
customers: one row per customer, with customer_id, Cluster, and raw engineered
    features. Never pass scaled/log-transformed model inputs as raw features.
    A custom column contract can be supplied with DNAConfig.
economics: optional one-row-per-customer DataFrame, joined by customer_id.
    Supported columns: gross_profit, servicing_cost, acquisition_cost,
    predicted_clv. Monetary values must use the SAME currency; costs/profit must
    cover the SAME observation window as spend. predicted_clv is supplied by a
    separate model, never calculated from historical spend by this panel.
transactions: optional transaction-line DataFrame with customer_id,
    transaction_id, transaction_date, price, quantity, and optional category.
    It must belong to the same source/window as customers. The caller must
    verify source identity; this presentation module cannot prove it.

Missing values remain missing; no zero-imputation. Full-base statistics use
all supplied customers, even when eligible_segments restricts the selector.
No disk writes, shared Streamlit cache, model training, or network calls occur
at runtime. Standalone uploads remain in memory. CSV exports are opt-in and
exclude customer identifiers. Data privacy/authentication remain host concerns.

Interpretation notes
--------------------
Total_Lifetime_Spend is observed spend, not predicted lifetime value.
Churn_Risk_Index_CRI = Recency_Days / Average_InterPurchase_Time_AIPT when
cadence > 0; the upstream pipeline uses 0 otherwise. It is not a probability.
Basket_Value_Variance and UPT_Variance are actually sample standard deviations
in the current repository. Inferred_Discount_Affinity is discounted LINE count
/ total units, not a conventional unit-weighted discount proportion.
Descriptive standardized differences are NOT significance tests, causal model
explanations, or cluster membership explanations.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Mapping, Sequence
import hashlib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DNAConfig:
    """Column mapping, monetary label, and descriptive trait threshold."""
    customer_col: str = "customer_id"
    segment_col: str = "Cluster"
    spend_col: str = "Total_Lifetime_Spend"
    orders_col: str = "Total_Lifetime_Transactions"
    currency: str = "CU"  # unspecified currency units, never infer from location
    trait_threshold: float = 0.25


@dataclass(frozen=True)
class FeatureSpec:
    label: str
    unit: str
    group: str
    definition: str


def spec(label, unit, group, definition):
    return FeatureSpec(label, unit, group, definition)


FEATURES = {
    "Total_Lifetime_Spend": spec("Observed spend", "money", "Economic value", "Sum of price × quantity over the observed window; not predicted CLV."),
    "Total_Lifetime_Units": spec("Observed units", "units", "Economic value", "Sum of purchased quantity over the observed window."),
    "Total_Lifetime_Transactions": spec("Order count", "orders", "Economic value", "Distinct transaction IDs per customer."),
    "Average_Order_Value_AOV": spec("Customer AOV", "money", "Economic value", "Customer observed spend / customer order count."),
    "Average_Unit_Retail_AUR": spec("Average unit retail", "money", "Economic value", "Customer spend / customer purchased units."),
    "Recency_Days": spec("Purchase recency", "days", "Cadence", "Days since last purchase, using the latest source date + 1 day as reference."),
    "Tenure_Days": spec("Observed tenure", "days", "Cadence", "Days between first and last observed purchase, not time since signup."),
    "Active_Days": spec("Active purchase days", "days", "Cadence", "Distinct calendar dates with purchases."),
    "Active_Days_Ratio": spec("Active days / tenure", "ratio", "Cadence", "Active days / tenure; upstream uses 1 for zero tenure. Not necessarily bounded by 1."),
    "Average_InterPurchase_Time_AIPT": spec("Purchase interval", "days", "Cadence", "Mean gap between distinct purchase dates; 0 when only one purchase date exists."),
    "Churn_Risk_Index_CRI": spec("Purchase delay / cadence", "ratio", "Cadence", "Recency / mean purchase interval; upstream 0 if interval is 0. Not a churn probability."),
    "Average_Units_Per_Transaction_UPT": spec("Units per order", "units/order", "Basket", "Purchased units / transaction count."),
    "Max_Basket_Value": spec("Largest basket", "money", "Basket", "Maximum transaction spend."),
    "Min_Basket_Value": spec("Smallest basket", "money", "Basket", "Minimum transaction spend."),
    "Basket_Value_Variance": spec("Basket spend std dev", "money", "Basket", "Upstream name says variance, but the implementation calculates sample standard deviation; singleton becomes 0."),
    "UPT_Variance": spec("Basket units std dev", "units", "Basket", "Sample standard deviation of units per basket, despite the upstream variance name; singleton becomes 0."),
    "Single_Item_Basket_Propensity": spec("Single-unit basket share", "fraction", "Basket", "Share of transactions whose summed quantity equals one."),
    "Category_Width": spec("Category breadth", "categories", "Breadth & price", "Distinct purchased categories."),
    "Department_Width": spec("Department breadth", "departments", "Breadth & price", "Distinct purchased departments."),
    "SKU_Breadth": spec("SKU breadth", "SKUs", "Breadth & price", "Distinct purchased product IDs."),
    "Department_Penetration_Pct": spec("Department penetration", "fraction", "Breadth & price", "Customer department breadth / source-wide department count."),
    "Category_Concentration_Score": spec("Top-category spend share", "fraction", "Breadth & price", "Largest category spend / customer spend; undefined at zero spend."),
    "Median_Item_Price": spec("Median line price", "money", "Breadth & price", "Median item price across transaction lines, not quantity weighted."),
    "Price_Variability_Score": spec("Line price std dev", "money", "Breadth & price", "Sample standard deviation of line prices; singleton becomes 0."),
    "Inferred_Discount_Affinity": spec("Discount-line / unit ratio", "ratio", "Breadth & price", "Count of lines priced below 80% of source-wide maximum SKU price / total units. This is a proxy, not verified promotion data or a unit-weighted share."),
    "gross_profit": spec("Observed gross profit", "money", "Economics supplied", "Supplied customer-level gross profit; not inferred from spend."),
    "servicing_cost": spec("Observed servicing cost", "money", "Economics supplied", "Supplied servicing cost for the same window as spend."),
    "acquisition_cost": spec("Observed acquisition cost", "money", "Economics supplied", "Supplied acquisition cost for the same window as spend."),
    "predicted_clv": spec("Supplied predicted CLV", "money", "Economics supplied", "External model estimate; its horizon, assumptions, and currency must be supplied by the caller."),
}
ECONOMIC_COLUMNS = ("gross_profit", "servicing_cost", "acquisition_cost", "predicted_clv")


def feature_catalog(config: DNAConfig) -> dict[str, FeatureSpec]:
    """Use explicit config mappings instead of guessing aliases."""
    catalog = FEATURES.copy()
    for canonical, actual in (("Total_Lifetime_Spend", config.spend_col),
                              ("Total_Lifetime_Transactions", config.orders_col)):
        if actual != canonical:
            catalog[actual] = catalog.pop(canonical)
    return catalog


def _ids(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name}: missing identifiers are not allowed.")
    normalized = values.astype(str).str.strip()
    if normalized.eq("").any():
        raise ValueError(f"{name}: blank identifiers are not allowed.")
    return normalized


def prepare_customers(customers: pd.DataFrame, config: DNAConfig,
                      economics: pd.DataFrame | None = None
                      ) -> tuple[pd.DataFrame, list[str]]:
    """Copy, validate one-row-per-customer grain, and sanitize known features."""
    if not isinstance(customers, pd.DataFrame) or customers.empty:
        raise ValueError("customers must be a non-empty Pandas DataFrame.")
    if not customers.columns.is_unique:
        raise ValueError("Duplicate customer column names are not allowed.")
    required = {config.customer_col, config.segment_col}
    if len(required) != 2:
        raise ValueError("Customer and segment columns must differ.")
    if not required.issubset(customers.columns):
        raise ValueError(f"Missing customer columns: {sorted(required - set(customers.columns))}")
    if config.trait_threshold < 0 or not np.isfinite(config.trait_threshold):
        raise ValueError("trait_threshold must be finite and nonnegative.")
    
    # Project to registered columns before deep-copying
    catalog = feature_catalog(config)
    keep = list(dict.fromkeys([
        config.customer_col,
        config.segment_col,
        *[c for c in catalog if c in customers.columns],
    ]))
    data = customers.loc[:, keep].copy(deep=True)
    
    data[config.customer_col] = _ids(data[config.customer_col], config.customer_col)
    if data[config.customer_col].duplicated().any():
        raise ValueError("Expected one row per customer; duplicate customer IDs found.")
    raw_segments = data[config.segment_col]
    normalized = _ids(raw_segments, config.segment_col)
    if raw_segments.nunique() != normalized.nunique():
        raise ValueError("Segment labels collide after string normalization.")
    data[config.segment_col] = normalized
    notes = []
    if economics is not None:
        if not isinstance(economics, pd.DataFrame) or not economics.columns.is_unique:
            raise ValueError("economics must be a DataFrame with unique columns.")
        if config.customer_col not in economics:
            raise ValueError("economics is missing the customer identifier column.")
        econ = economics.copy(deep=True)
        econ[config.customer_col] = _ids(econ[config.customer_col], config.customer_col)
        if econ[config.customer_col].duplicated().any():
            raise ValueError("economics must have one row per customer.")
        cols = [c for c in ECONOMIC_COLUMNS if c in econ]
        if any(c in data for c in cols):
            raise ValueError("Economic fields already exist in customers; do not supply them twice.")
        unknown = (~econ[config.customer_col].isin(data[config.customer_col])).sum()
        if unknown:
            notes.append(f"Ignored {unknown:,} economic rows outside the customer base.")
        data = data.merge(econ[[config.customer_col] + cols], how="left",
                          on=config.customer_col, validate="one_to_one")
    for name, metadata in feature_catalog(config).items():
        if name not in data:
            continue
        original = data[name]
        numeric = pd.to_numeric(original, errors="coerce").astype(float)
        invalid = original.notna() & ~np.isfinite(numeric)
        if invalid.any():
            notes.append(f"{name}: {invalid.sum():,} nonnumeric/nonfinite values treated as missing.")
        data[name] = numeric.replace([np.inf, -np.inf], np.nan)
        if metadata.group != "Economics supplied" and data[name].lt(0).any():
            raise ValueError(f"{name} contains negative values incompatible with these feature definitions.")
        if metadata.unit == "fraction" and data[name].gt(1 + 1e-8).any():
            raise ValueError(f"{name} must be a fraction in [0, 1], not a percentage in [0, 100].")
    if not any(c in data and data[c].notna().any() for c in feature_catalog(config)):
        raise ValueError("No usable registered features found. Check column mappings and raw numeric values.")
    return data, notes


def summarize_features(data: pd.DataFrame, segment: str, config: DNAConfig) -> pd.DataFrame:
    """Raw summaries plus (segment mean - full-base mean) / full-base SD.

    Population standard deviation (ddof=0) is used consistently. Zero variance
    yields an undefined z difference, not a fabricated zero. Missing values are
    excluded per feature; valid sample sizes are always available in the table.
    """
    catalog = feature_catalog(config)
    features = [c for c in catalog if c in data.columns]

    base = data[features]
    chosen = data.loc[
        data[config.segment_col].eq(str(segment)), features
    ]
    quantiles = chosen.quantile([0.25, 0.75])
    sd = base.std(ddof=0)

    out = pd.DataFrame({
        "segment_n": chosen.count(),
        "base_n": base.count(),
        "segment_mean": chosen.mean(),
        "base_mean": base.mean(),
        "segment_median": chosen.median(),
        "base_median": base.median(),
        "segment_p25": quantiles.loc[0.25],
        "segment_p75": quantiles.loc[0.75],
        "base_sd": sd,
    })

    denominator = sd.where(np.isfinite(sd) & sd.gt(0))
    out["z_difference"] = (
        (out["segment_mean"] - out["base_mean"]) / denominator
    ).replace([np.inf, -np.inf], np.nan)

    out.index.name = "feature"
    out = out.reset_index()

    for field in ("label", "group", "unit", "definition"):
        out[field] = out["feature"].map(
            lambda name: getattr(catalog[name], field)
        )
    return out


def contribution_table(data: pd.DataFrame, config: DNAConfig) -> pd.DataFrame:
    """Stable full-base customer/spend denominators and explicit coverage."""
    grouped = data.groupby(config.segment_col, sort=True)
    out = grouped.size().rename("customers").to_frame()
    out["customer_share"] = out.customers / len(data)
    if config.spend_col in data:
        out["spend_valid_n"] = grouped[config.spend_col].count()
        out["observed_spend"] = grouped[config.spend_col].sum(min_count=1)
        total = data[config.spend_col].sum(min_count=1)
        out["spend_share"] = out.observed_spend / total if pd.notna(total) and total > 0 else np.nan
        out["contribution_index"] = out.spend_share / out.customer_share
        out["share_gap_pp"] = (out.spend_share - out.customer_share) * 100
    return out


def prepare_transactions(transactions: pd.DataFrame, data: pd.DataFrame,
                         config: DNAConfig) -> tuple[pd.DataFrame, list[str]]:
    """Validate lines without silently deduplicating legitimate repeated lines."""
    required = {config.customer_col, "transaction_id", "transaction_date", "price", "quantity"}
    if not isinstance(transactions, pd.DataFrame) or not transactions.columns.is_unique:
        raise ValueError("transactions must be a DataFrame with unique columns.")
    if not required.issubset(transactions):
        raise ValueError(f"Missing transaction columns: {sorted(required - set(transactions.columns))}")
    
    # Project to registered columns before deep-copying
    keep = [
        config.customer_col,
        "transaction_id",
        "transaction_date",
        "price",
        "quantity",
    ]
    if "category" in transactions.columns:
        keep.append("category")
    tx = transactions.loc[:, keep].copy(deep=True)
    
    ids = tx[config.customer_col].astype("string").str.strip()
    order_ids = tx.transaction_id.astype("string").str.strip()
    tx[config.customer_col] = ids
    tx["transaction_id"] = order_ids
    tx["transaction_date"] = pd.to_datetime(tx.transaction_date, errors="coerce", utc=True).dt.tz_convert(None)
    for c in ("price", "quantity"):
        tx[c] = pd.to_numeric(tx[c], errors="coerce").astype(float)
    valid = (ids.notna() & ids.ne("") & order_ids.notna() & order_ids.ne("")
             & tx.transaction_date.notna() & np.isfinite(tx.price) & tx.price.ge(0)
             & np.isfinite(tx.quantity) & tx.quantity.gt(0)).fillna(False)
    notes = []
    if (~valid).any():
        notes.append(f"Excluded {(~valid).sum():,} invalid transaction lines.")
    tx = tx.loc[valid].copy()
    known = tx[config.customer_col].isin(data[config.customer_col])
    if (~known).any():
        notes.append(f"Excluded {(~known).sum():,} transaction lines outside the customer base.")
    tx = tx.loc[known].merge(data[[config.customer_col, config.segment_col]],
                            on=config.customer_col, validate="many_to_one")
    tx["line_spend"] = tx.price * tx.quantity
    finite = np.isfinite(tx.line_spend)
    if (~finite).any():
        notes.append(f"Excluded {(~finite).sum():,} lines with overflowing price × quantity.")
    tx = tx.loc[finite].copy()
    return tx, notes


def format_value(value, unit="", currency="CU") -> str:
    if pd.isna(value) or not np.isfinite(value):
        return "N/A"
    if unit == "money":
        return f"{currency} {value:,.2f}"
    if unit == "fraction":
        return f"{value:.1%}"
    if unit == "ratio":
        return f"{value:,.2f}×"
    return f"{value:,.2f}" + (f" {unit}" if unit else "")


def _chart(st, figure, key):
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(figure, use_container_width=True, key=key)


def render_segment_dna(customers: pd.DataFrame, *,
                       economics: pd.DataFrame | None = None,
                       transactions: pd.DataFrame | None = None,
                       config: DNAConfig | None = None,
                       eligible_segments: Sequence | None = None,
                       segment_labels: Mapping | None = None,
                       key_prefix: str = "dna") -> str | None:
    """Render into the current Streamlit container; return selected segment ID.

    Full customers must be passed, not the host's prefiltered view. Selector
    eligibility is separate, so benchmarks do not change with global filters.
    UI imports are lazy to allow headless tests of all analysis functions.
    Optional economics/transactions are validated independently; their errors
    disable only their respective views. Main input errors return None.
    """
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    cfg = config or DNAConfig()
    st.header("Segment DNA")
    try:
        data, notes = prepare_customers(customers, cfg)
    except (ValueError, TypeError) as error:
        st.error(str(error))
        return None
    if economics is not None:
        try:
            data, economic_notes = prepare_customers(customers, cfg, economics)
            notes.extend(economic_notes)
        except (ValueError, TypeError) as error:
            st.warning(f"Optional economics unavailable: {error}")
    options = sorted(data[cfg.segment_col].unique().tolist())
    if eligible_segments is not None:
        eligible = {str(x).strip() for x in eligible_segments}
        options = [x for x in options if x in eligible]
    if not options:
        st.info("No segments are eligible under the current global filters.")
        return None
    labels = {str(k).strip(): str(v) for k, v in (segment_labels or {}).items()}
    dataset_key = hashlib.sha256(pd.util.hash_pandas_object(
        data[[cfg.customer_col, cfg.segment_col]], index=False).values.tobytes()).hexdigest()[:12]
    prefix = f"{key_prefix}_{dataset_key}"
    select_key = f"{prefix}_segment"
    if select_key in st.session_state and st.session_state[select_key] not in options:
        del st.session_state[select_key]
    segment = st.selectbox(
        "Select a customer segment",
        options,
        key=select_key,
        format_func=lambda x: f"{labels.get(x, 'Segment ' + x)} · ID {x}",
        help=(
            "Global filters control which segments you can select. "
            "Means, standard deviations and share denominators "
            "always use the full customer base."
        ),
    )
    subset = data[data[cfg.segment_col].eq(segment)]
    with st.spinner("Calculating segment characteristics..."):
        summary = summarize_features(data, segment, cfg)
        contribution = contribution_table(data, cfg)
    row = contribution.loc[segment]
    st.subheader(labels.get(segment, f"Segment {segment}"))
    st.caption(f"{len(subset):,} customers in this segment; {len(data):,} in the full base. "
               "Benchmarks and share denominators remain unfiltered. All behavior is descriptive.")
    cards = st.columns(4)
    cards[0].metric("Customers", f"{len(subset):,}", f"{row.customer_share:.1%} of full base", delta_color="off")
    spend = row.get("observed_spend", np.nan)
    spend_share = row.get("spend_share", np.nan)
    cards[1].metric("Observed spend", format_value(spend, "money", cfg.currency),
                    f"{spend_share:.1%} of known spend" if pd.notna(spend_share) else None, delta_color="off")
    index = row.get("contribution_index", np.nan)
    cards[2].metric("Spend contribution index", format_value(index, "ratio"))
    orders = subset[cfg.orders_col].sum(min_count=1) if cfg.orders_col in subset else np.nan
    # Weighted AOV uses paired spend/order coverage, never mismatched sums.
    aov = np.nan
    if {cfg.spend_col, cfg.orders_col}.issubset(subset):
        paired = subset[[cfg.spend_col, cfg.orders_col]].dropna()
        denominator = paired[cfg.orders_col].sum()
        if denominator > 0:
            aov = paired[cfg.spend_col].sum() / denominator
    cards[3].metric("Order-weighted AOV", format_value(aov, "money", cfg.currency))
    st.caption("Contribution index = known-spend share / customer share; 1× means proportional contribution. "
               "Order-weighted AOV = paired customers' total spend / total orders, not mean customer AOV.")
    if cfg.spend_col not in data:
        st.warning("Observed spend column is absent; monetary contribution metrics are unavailable.")
    elif data[cfg.spend_col].isna().any():
        st.warning(f"Spend coverage: {data[cfg.spend_col].notna().sum():,}/{len(data):,} full-base customers; "
                   f"{subset[cfg.spend_col].notna().sum():,}/{len(subset):,} selected customers. "
                   "Shares use known spend only and can be biased by missingness.")
    if len(subset) < 30:
        st.warning("Small segment (fewer than 30 customers): descriptive summaries may be unstable.")
    
    # Feature-level coverage warning
    coverage = summary["segment_n"] / len(subset)
    low_coverage = summary.loc[
        coverage.lt(0.8),
        ["label", "segment_n"],
    ]
    if not low_coverage.empty:
        st.warning(
            f"{len(low_coverage)} features have less than 80% coverage "
            "in this segment. Interpret their differences cautiously."
        )
        with st.expander("Features with incomplete coverage"):
            st.dataframe(
                low_coverage.assign(
                    coverage=coverage.loc[low_coverage.index]
                ),
                hide_index=True,
                use_container_width=True,
            )
    
    if notes:
        with st.expander("Input quality notes"):
            for note in dict.fromkeys(notes):
                st.write(note)

    traits = summary.dropna(subset=["z_difference"]).copy()
    traits = traits[traits.z_difference.abs().ge(cfg.trait_threshold)]
    traits = traits.loc[traits.z_difference.abs().sort_values(ascending=False).index].head(3)
    st.subheader("Distinctive observed traits")
    if traits.empty:
        st.info("No available feature crosses the configured standardized-difference threshold.")
    else:
        for _, trait in traits.iterrows():
            direction = "higher" if trait.z_difference > 0 else "lower"
            st.write(f"{trait.label}: {abs(trait.z_difference):.2f} full-base SD {direction}. "
                     f"Segment mean {format_value(trait.segment_mean, trait.unit, cfg.currency)}; "
                     f"base mean {format_value(trait.base_mean, trait.unit, cfg.currency)} "
                     f"(valid n={int(trait.segment_n):,}/{int(trait.base_n):,}).")
    st.caption("Traits rank observed differences, not causes, statistical significance, or predictions. "
               "Higher is not automatically better; longer recency and purchase intervals describe slower activity.")

    overview, behavior, economics_tab, purchases, definitions = st.tabs(
        ["DNA overview", "Within-segment variation", "Economic value", "Purchase mix & trend", "Definitions & quality"])
    with overview:
        groups = summary.group.unique().tolist()
        chosen_groups = st.multiselect("Feature families", groups, default=groups, key=f"{prefix}_groups")
        
        show_all = st.checkbox(
            "Show all available features",
            value=False,
            key=f"{prefix}_all_features",
            help="By default, show the eight largest differences from the full base.",
        )

        chart_data = summary.loc[
            summary["group"].isin(chosen_groups)
        ].dropna(subset=["z_difference"]).copy()

        if not show_all:
            chart_data = chart_data.loc[
                chart_data["z_difference"].abs().nlargest(8).index
            ]

        chart_data = chart_data.sort_values("z_difference")

        chart_col, guide_col = st.columns([3, 1])

        with chart_col:
            if chart_data.empty:
                st.info("Select a feature family with observed, nonconstant features.")
            else:
                limit = max(0.5, float(chart_data["z_difference"].abs().max()))
                fig = px.bar(
                    chart_data,
                    x="z_difference",
                    y="label",
                    orientation="h",
                    color="z_difference",
                    color_continuous_scale="RdBu",
                    range_color=(-limit, limit),
                    hover_data=[
                        "segment_mean", "base_mean", "unit", "segment_n", "base_n"
                    ],
                    labels={"z_difference": "Difference from full base (SD)", "label": ""},
                )
                fig.add_vline(x=0, line_color="#64748b")
                fig.update_layout(
                    height=max(360, 32 * len(chart_data)),
                    coloraxis_showscale=False,
                )
                _chart(st, fig, f"{prefix}_dna_bar")

        with guide_col:
            st.info(
                "Read the profile\n\n"
                "• Right: higher than the full-base mean.\n"
                "• Left: lower than the full-base mean.\n"
                "• Zero: equal to the full-base mean.\n\n"
                "Differences describe behavior—not causes or segment quality."
            )

        with st.expander("Exact values, quartiles and valid sample sizes"):
            st.dataframe(
                summary[[
                    "label", "segment_n", "base_n",
                    "segment_mean", "base_mean",
                    "segment_median", "segment_p25",
                    "segment_p75", "z_difference",
                ]],
                hide_index=True,
                use_container_width=True,
            )
        
        st.download_button("Download aggregate DNA (CSV)", summary.to_csv(index=False).encode("utf-8"),
                           file_name="segment_dna_summary.csv", mime="text/csv", key=f"{prefix}_export")

        # Optional cluster-comparison heatmap
        if st.checkbox(
            "Compare eligible segments",
            key=f"{prefix}_compare_segments",
            help="Each cell compares a segment mean with the same full-base benchmark.",
        ):
            features = summary.loc[
                summary["group"].isin(chosen_groups), "feature"
            ].tolist()

            if not features:
                st.info("Select at least one feature family.")
            else:
                base_sd = data[features].std(ddof=0)
                base_sd = base_sd.where(np.isfinite(base_sd) & base_sd.gt(0))

                means = data.groupby(
                    cfg.segment_col, observed=True, sort=False
                )[features].mean().reindex(options)

                z = means.sub(data[features].mean()).div(base_sd).dropna(
                    axis=1, how="all"
                )

                if z.empty:
                    st.info("No observed, nonconstant features to compare.")
                else:
                    catalog = feature_catalog(cfg)
                    fig = go.Figure(go.Heatmap(
                        z=z.to_numpy(),
                        x=[catalog[c].label for c in z.columns],
                        y=[f"{labels.get(s, 'Segment ' + s)} · ID {s}" for s in z.index],
                        colorscale="RdBu",
                        zmid=0,
                        hoverongaps=False,
                        colorbar=dict(title="Base SD"),
                    ))
                    fig.update_layout(
                        height=max(320, 45 * len(z)),
                        xaxis=dict(tickangle=-45),
                    )
                    _chart(st, fig, f"{prefix}_comparison_heatmap")

    with behavior:
        available = summary.loc[summary.segment_n.gt(0), "feature"].tolist()
        
        metric_key = f"{prefix}_distribution_metric"

        if (
            metric_key in st.session_state
            and st.session_state[metric_key] not in available
        ):
            del st.session_state[metric_key]

        inspect_distribution = st.checkbox(
            "Load within-segment distribution",
            key=f"{prefix}_load_distribution",
            help="Loads the customer-level boxplot and detailed statistics.",
        )
        
        if inspect_distribution:
            if available:
                metric = st.selectbox(
                    "Feature to inspect",
                    available,
                    key=metric_key,
                    format_func=lambda x: feature_catalog(cfg)[x].label,
                    help="Compare this segment with customers outside it.",
                )
                meta = feature_catalog(cfg)[metric]
                log_axis = st.checkbox("Logarithmic value axis", key=f"{prefix}_log")
                distribution = pd.DataFrame({"value": data[metric], "comparison": np.where(
                    data[cfg.segment_col].eq(segment), "Selected segment", "Other customers")}).dropna()
                if log_axis:
                    omitted = int(distribution.value.le(0).sum())
                    distribution = distribution[distribution.value.gt(0)]
                    st.caption(f"{omitted:,} non-positive observations omitted from the chart only; summary statistics keep them.")
                if distribution.empty:
                    st.info(
                        "No values remain for this chart. "
                        "Disable the logarithmic axis or choose another feature."
                    )
                else:
                    counts = distribution["comparison"].value_counts()
                    if counts.get("Other customers", 0) == 0:
                        st.info(
                            "No observed values remain outside this segment; "
                            "the chart shows the selected segment only."
                        )
                    fig = px.box(distribution, x="comparison", y="value", color="comparison", points=False,
                                 log_y=log_axis, labels={"value": f"{meta.label} ({cfg.currency if meta.unit == 'money' else meta.unit})",
                                                   "comparison": ""},
                                 title="Selected segment versus non-overlapping remainder")
                    _chart(st, fig, f"{prefix}_distribution")
                # Narrow temporary frame for distribution statistics
                stats = data[[metric]].copy()
                stats["comparison"] = np.where(
                    data[cfg.segment_col].eq(segment),
                    "Selected segment",
                    "Other customers",
                )
                st.dataframe(
                    stats.groupby("comparison", sort=False)[metric].agg(
                        ["count", "mean", "median", "std", "min", "max"]
                    ),
                    use_container_width=True,
                )
                st.caption(meta.definition + " Boxplot compares disjoint groups; the DNA benchmark includes the entire base. Table std uses sample SD.")
            else:
                st.info("This segment has no observed feature values to inspect.")
        else:
            st.info("Enable this option to inspect customer-level variation.")

    with economics_tab:
        shares = contribution.reset_index()
        if "spend_share" in shares:
            long = shares.melt(id_vars=[cfg.segment_col], value_vars=["customer_share", "spend_share"],
                               var_name="share", value_name="fraction")
            fig = px.bar(long, x=cfg.segment_col, y="fraction", color="share", barmode="group",
                         title="Full-base customer share versus known-spend share")
            fig.update_yaxes(tickformat=".0%")
            _chart(st, fig, f"{prefix}_shares")
        st.dataframe(contribution, use_container_width=True)
        econ_rows = []
        for name in ECONOMIC_COLUMNS:
            if name in subset:
                series = subset[name]
                econ_rows.append({"metric": FEATURES[name].label, "valid_customers": int(series.count()),
                                  "coverage": series.count() / len(subset), "sum_known": series.sum(min_count=1),
                                  "mean_known": series.mean(), "currency": cfg.currency})
        if econ_rows:
            st.dataframe(pd.DataFrame(econ_rows), use_container_width=True, hide_index=True)
            st.caption("Economic sums include only observed values, with coverage shown. "
                       "Predicted CLV is a separate supplied estimate; its forecast horizon is not the historical window.")
        else:
            st.info("No costs, profit, or predicted CLV supplied. Spend alone cannot establish profitability or lifetime value.")
        if "gross_profit" in subset:
            cost_cols = [c for c in ("servicing_cost", "acquisition_cost") if c in subset]
            if cost_cols:
                complete = subset[["gross_profit"] + cost_cols].dropna()
                contribution_margin = complete.gross_profit - complete[cost_cols].sum(axis=1)
                st.metric("Gross profit minus supplied costs (known rows)",
                          format_value(contribution_margin.sum(min_count=1), "money", cfg.currency))
                st.caption(f"Complete-row coverage {len(complete):,}/{len(subset):,}; deducted: {', '.join(cost_cols)}. "
                           "This is not net profit; omitted overhead or other costs are not inferred.")

    with purchases:
        if transactions is None:
            st.info("Supply transactions to enable category mix and monthly observed-spend trends.")
        else:
            try:
                tx, tx_notes = prepare_transactions(transactions, data, cfg)
                for note in tx_notes:
                    st.warning(note)
                picked_tx = tx[tx[cfg.segment_col].eq(segment)].copy()
                if picked_tx.empty:
                    st.info("No valid transaction lines for this segment.")
                else:
                    if cfg.spend_col in subset and subset[cfg.spend_col].notna().all():
                        expected, actual = subset[cfg.spend_col].sum(), picked_tx.line_spend.sum()
                        if not np.isclose(expected, actual, rtol=1e-6, atol=.01):
                            st.warning("Transaction spend does not reconcile with engineered segment spend. "
                                       "Check source identity, observation window, currency, and upstream validation.")
                    st.caption(f"Transaction window (UTC-normalized dates): {tx.transaction_date.min().date()} "
                               f"to {tx.transaction_date.max().date()}. Same segment membership is applied across all dates.")
                    start = tx.transaction_date.min().to_period("M")
                    end = tx.transaction_date.max().to_period("M")
                    monthly = picked_tx.assign(month=picked_tx.transaction_date.dt.to_period("M")).groupby("month").line_spend.sum()
                    monthly = monthly.reindex(pd.period_range(start, end, freq="M"), fill_value=0)
                    trend = monthly.rename("observed_spend").rename_axis("month").reset_index()
                    trend["month"] = trend.month.astype(str)
                    fig = px.line(trend, x="month", y="observed_spend", markers=True,
                                  labels={"observed_spend": f"Observed spend ({cfg.currency})"},
                                  title="Monthly observed spend — historical segment membership")
                    _chart(st, fig, f"{prefix}_trend")
                    st.caption("Zero indicates no observed lines in a source-window month, not confirmed inactivity. "
                               "First/last months may be partial; data completeness is not inferred. This is not retention.")
                    if "category" in picked_tx:
                        picked_tx["category"] = picked_tx.category.fillna("Unknown").astype(str)
                        mix = picked_tx.groupby("category").line_spend.sum().sort_values(ascending=False)
                        total = mix.sum()
                        if total > 0:
                            top = mix.head(10)
                            mix_data = pd.DataFrame({"category": top.index, "spend": top.values,
                                                     "segment_spend_share": top.values / total})
                            fig = px.bar(mix_data, x="segment_spend_share", y="category", orientation="h",
                                         hover_data=["spend"], title="Top categories by observed spend")
                            fig.update_xaxes(tickformat=".0%")
                            fig.update_yaxes(categoryorder="total ascending")
                            _chart(st, fig, f"{prefix}_categories")
                            st.caption(f"Top {len(top)} of {len(mix)} categories; shares use all-category spend. "
                                       f"Categories not drawn account for {(total - top.sum()) / total:.1%}.")
            except (ValueError, TypeError) as error:
                st.warning(f"Transaction views unavailable: {error}")

    with definitions:
        st.dataframe(summary[["feature", "label", "unit", "definition", "segment_n", "base_n"]],
                     use_container_width=True, hide_index=True)
        missing = pd.DataFrame({"feature": summary.feature, "segment_missing": len(subset) - summary.segment_n,
                                "base_missing": len(data) - summary.base_n})
        st.dataframe(missing, use_container_width=True, hide_index=True)
        if "Average_InterPurchase_Time_AIPT" in subset:
            no_cadence = subset.Average_InterPurchase_Time_AIPT.le(0).sum()
            st.caption(f"{no_cadence:,} selected customers have zero/nonpositive purchase interval. "
                       "Their upstream CRI of zero does not establish low churn risk.")
        st.caption("Panel uses registered raw features only. PC coordinates, identifiers, arbitrary numeric codes, "
                   "and unregistered fields are not silently treated as behavioral DNA. Currency must be specified explicitly.")
    return segment


def make_demo_data(seed: int = 42, n: int = 600) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproducible synthetic customers, reconciled transactions, and economics."""
    if n < 1:
        raise ValueError("Demo customer count must be positive.")
    rng = np.random.default_rng(seed)
    rows, lines = [], []
    reference = pd.Timestamp("2026-01-01")
    categories = np.array(["Home", "Food", "Beauty", "Electronics", "Clothing"])
    for i in range(n):
        cluster = i % 3
        cid = f"DEMO-{i:05d}"
        count = int(rng.integers(*[(10, 25), (3, 10), (1, 5)][cluster]))
        recency = int(rng.integers(*[(1, 20), (15, 70), (65, 140)][cluster]))
        dates = pd.date_range(end=reference - pd.Timedelta(days=recency), periods=count,
                              freq=f"{int(rng.integers(5, 18))}D")
        units = rng.integers(1, 7, count)
        prices = rng.lognormal(np.log([45, 20, 12][cluster]), .4, count).round(2)
        cats = rng.choice(categories, count)
        values = units * prices
        spend = values.sum()
        tenure = (dates[-1] - dates[0]).days
        cadence = tenure / (count - 1) if count > 1 else 0
        for j in range(count):
            lines.append(dict(customer_id=cid, transaction_id=f"{cid}-{j}",
                              transaction_date=dates[j], price=prices[j], quantity=int(units[j]), category=cats[j]))
        category_spend = pd.Series(values).groupby(cats).sum()
        rows.append(dict(customer_id=cid, Cluster=cluster, Total_Lifetime_Spend=spend,
                         Total_Lifetime_Units=units.sum(), Total_Lifetime_Transactions=count,
                         Average_Order_Value_AOV=spend / count, Average_Unit_Retail_AUR=spend / units.sum(),
                         Recency_Days=recency, Tenure_Days=tenure, Active_Days=count,
                         Average_InterPurchase_Time_AIPT=cadence,
                         Churn_Risk_Index_CRI=recency / cadence if cadence > 0 else 0,
                         Average_Units_Per_Transaction_UPT=units.mean(), Max_Basket_Value=values.max(),
                         Min_Basket_Value=values.min(), Basket_Value_Variance=values.std(ddof=1) if count > 1 else 0,
                         Single_Item_Basket_Propensity=(units == 1).mean(), Category_Width=len(set(cats)),
                         Category_Concentration_Score=category_spend.max() / spend,
                         Median_Item_Price=np.median(prices), Price_Variability_Score=prices.std(ddof=1) if count > 1 else 0))
    customers = pd.DataFrame(rows)
    economics = pd.DataFrame({"customer_id": customers.customer_id,
                             "gross_profit": customers.Total_Lifetime_Spend * rng.uniform(.18, .4, n),
                             "servicing_cost": customers.Total_Lifetime_Transactions * 2.5,
                             "acquisition_cost": rng.uniform(10, 35, n)})
    return customers, economics, pd.DataFrame(lines)


def read_uploaded_table(upload, customer_col: str) -> pd.DataFrame:
    """CSV IDs are kept as strings (including leading zeros); Parquet needs pyarrow."""
    payload = BytesIO(upload.getvalue())
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(payload, dtype={customer_col: "string", "transaction_id": "string"})
    return pd.read_parquet(payload)


def main() -> None:
    import streamlit as st
    st.set_page_config(page_title="Customer Segment DNA", layout="wide")
    st.title("Customer Segment DNA")
    st.sidebar.header("Input data")
    currency = st.sidebar.text_input("Currency code / monetary unit", value="CU").strip() or "CU"
    demo = st.sidebar.checkbox("Use synthetic demo data", value=True)
    if demo:
        customers, economics, transactions = make_demo_data()
        st.warning("SYNTHETIC DEMO — generated examples, not your repository's customer data. Monetary values are illustrative.")
    else:
        st.sidebar.caption("Use anonymized data in public deployments. Uploads are processed in memory without a shared result cache.")
        upload = st.sidebar.file_uploader("Customer features and segment assignments", type=["csv", "parquet"], key="dna_customer_upload")
        economic_upload = st.sidebar.file_uploader("Optional customer economics", type=["csv", "parquet"], key="dna_economic_upload")
        transaction_upload = st.sidebar.file_uploader("Optional transaction lines", type=["csv", "parquet"], key="dna_transaction_upload")
        if upload is None:
            st.info("Upload a customer-level table with customer_id, Cluster, and engineered features. This panel does not train segments.")
            return
        try:
            customers = read_uploaded_table(upload, "customer_id")
            economics = read_uploaded_table(economic_upload, "customer_id") if economic_upload else None
            transactions = read_uploaded_table(transaction_upload, "customer_id") if transaction_upload else None
        except Exception as error:
            st.error(f"Unable to read input table: {error}")
            return
    render_segment_dna(customers, economics=economics, transactions=transactions,
                       config=DNAConfig(currency=currency))


if __name__ == "__main__":
    main()
