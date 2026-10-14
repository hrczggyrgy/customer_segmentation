"""Standalone Streamlit UI for the 30-day purchase inactivity model."""
from __future__ import annotations

import hashlib
import io

import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st

from purchase_inactivity import (
    FEATURE_COLUMNS,
    explain_customer,
    fit_evaluate_and_score,
    global_feature_effects,
    prepare_transactions,
)

st.set_page_config(page_title="30-Day Purchase Inactivity", layout="wide")
st.title("30-Day Purchase Inactivity Model")
st.caption(
    "A lightweight, explainable baseline that scores customers who purchased in the last 30 days "
    "for the likelihood of making no purchase in the following 30 days."
)

st.info(
    "This is a probabilistic ranking, not a guarantee. The positive class is **no purchase in the next 30 days**. "
    "A predicted inactivity score is not a diagnosis of churn, and explanations describe model associations—not causes."
)

uploaded = st.sidebar.file_uploader("Upload cleaned transaction Parquet", type=["parquet"])
if uploaded is None:
    st.info("Upload the same cleaned transaction Parquet used by the segmentation app to begin.")
    st.stop()

try:
    raw_bytes = uploaded.getvalue()
    tx = pl.read_parquet(io.BytesIO(raw_bytes)).to_pandas()
    prepared = prepare_transactions(tx)
except Exception as exc:
    st.error(f"Could not read or validate this transaction file: {exc}")
    st.stop()

latest_transaction_date = prepared["order_date"].max().date()
st.caption(
    f"Valid transaction lines: {len(prepared):,} | Customers: {prepared['customer_id'].nunique():,} | "
    f"Latest observed transaction date: {latest_transaction_date.isoformat()}"
)

st.sidebar.subheader("Data coverage")
st.sidebar.warning(
    "The latest transaction date does not prove that the export is complete through that day. "
    "Confirm the source extract's actual coverage date before generating labels or scores."
)
coverage_confirmed = st.sidebar.checkbox(
    "I confirm the export is complete through the date below.", value=False
)
coverage_date = st.sidebar.date_input(
    "Data complete through",
    value=latest_transaction_date,
    min_value=latest_transaction_date,
    help="Use the date confirmed by the source system, not merely the last date on which a purchase occurred.",
)

if not coverage_confirmed:
    st.warning("Confirm data completeness in the sidebar to enable training and scoring.")
    st.stop()

file_fingerprint = hashlib.sha256(raw_bytes).hexdigest()
run_key = (file_fingerprint, coverage_date.isoformat())
if st.session_state.get("purchase_inactivity_error_key") != run_key:
    st.session_state.pop("purchase_inactivity_error", None)
    st.session_state.pop("purchase_inactivity_error_key", None)
if st.sidebar.button("Train model and score active customers", type="primary", use_container_width=True):
    with st.spinner("Building historical snapshots, testing the model, and scoring active customers…"):
        try:
            result = fit_evaluate_and_score(tx, coverage_date)
            st.session_state["purchase_inactivity_result"] = result
            st.session_state["purchase_inactivity_run_key"] = run_key
            st.session_state.pop("purchase_inactivity_error", None)
            st.session_state.pop("purchase_inactivity_error_key", None)
        except Exception as exc:
            st.session_state["purchase_inactivity_error"] = str(exc)
            st.session_state["purchase_inactivity_error_key"] = run_key
            st.session_state.pop("purchase_inactivity_result", None)
            st.session_state.pop("purchase_inactivity_run_key", None)

if (st.session_state.get("purchase_inactivity_error")
        and st.session_state.get("purchase_inactivity_error_key") == run_key):
    st.error(st.session_state["purchase_inactivity_error"])
    st.stop()

result = st.session_state.get("purchase_inactivity_result")
if result is None or st.session_state.get("purchase_inactivity_run_key") != run_key:
    st.info("Choose a confirmed completeness date, then click **Train model and score active customers**.")
    st.stop()

evaluation = result["evaluation"]
st.subheader("Out-of-time validation")
st.caption(
    "The latest historical snapshot windows are held out for evaluation; the model is not randomly splitting "
    "rows across time. This better approximates scoring customers after an earlier training period."
)
metric_cols = st.columns(4)
metric_cols[0].metric("Historical snapshots", f"{evaluation['n_snapshots']:,}")
metric_cols[1].metric("Holdout rows", f"{evaluation['n_test']:,}")
metric_cols[2].metric(
    "ROC-AUC",
    f"{evaluation['roc_auc']:.3f}" if evaluation["roc_auc"] is not None else "N/A",
    help="Measures ranking separation between the two classes; N/A when the holdout contains only one class.",
)
metric_cols[3].metric(
    "Average precision",
    f"{evaluation['average_precision']:.3f}" if evaluation["average_precision"] is not None else "N/A",
    help="Precision-recall summary for the no-purchase class; N/A when the holdout contains only one class.",
)
st.caption(
    f"Holdout snapshot dates: {evaluation['test_start']} through {evaluation['test_end']} | "
    f"No-purchase share in holdout: {evaluation['test_no_purchase_rate']:.1%} | "
    f"Brier score: {evaluation['brier_score']:.3f} (lower is better; probability calibration has not been independently validated)."
)
if evaluation["n_snapshots"] < 6:
    st.warning("There are relatively few historical snapshots. Treat holdout metrics as preliminary; more complete transaction history will make evaluation more reliable.")
if not evaluation["holdout_has_both_classes"]:
    st.warning("The time-based holdout contains only one target class, so ROC-AUC and average precision cannot be reported reliably for this split.")

predictions = result["predictions"]
current_features = result["current_features"]
st.subheader("Currently active customers ranked by inactivity score")
st.caption(
    f"Scored {len(predictions):,} customers with at least one purchase in the 30 days ending "
    f"{coverage_date.isoformat()}. Scores are model-estimated probabilities, not certainties."
)
if predictions.empty:
    st.warning("No customers had a purchase within the last 30 days of the confirmed coverage date.")
    st.stop()

risk_col, active_col = st.columns(2)
with risk_col:
    st.metric("Customers scored", f"{len(predictions):,}")
with active_col:
    st.metric("Mean estimated no-purchase probability", f"{predictions['no_purchase_probability'].mean():.1%}")

hist = px.histogram(
    predictions,
    x="no_purchase_probability",
    nbins=20,
    labels={"no_purchase_probability": "Estimated probability of no purchase next 30 days"},
    title="Distribution of inactivity scores",
)
hist.update_layout(height=330, yaxis_title="Customers")
st.plotly_chart(hist, use_container_width=True)

shown = predictions.head(100).copy()
shown["no_purchase_probability"] = shown["no_purchase_probability"].map(lambda x: f"{x:.1%}")
feature_lookup = current_features.set_index("customer_id")
for col in ["orders_30d", "spend_30d", "orders_90d", "spend_90d", "days_since_last_purchase"]:
    if col in feature_lookup:
        shown[col] = shown["customer_id"].map(feature_lookup[col])
shown = shown.rename(columns={
    "customer_id": "Customer ID",
    "no_purchase_probability": "Estimated no-purchase probability",
    "orders_30d": "Orders (30d)",
    "spend_30d": "Spend (30d)",
    "orders_90d": "Orders (90d)",
    "spend_90d": "Spend (90d)",
    "days_since_last_purchase": "Days since last purchase",
})
st.dataframe(shown, use_container_width=True, hide_index=True)
st.download_button(
    "Download all scored customers (CSV)",
    data=predictions.to_csv(index=False).encode("utf-8"),
    file_name="purchase_inactivity_scores.csv",
    mime="text/csv",
)

st.subheader("Why did the model give this score?")
customer_ids = predictions["customer_id"].astype(str).tolist()
selected_customer = st.selectbox("Choose a customer to explain", customer_ids)
local = explain_customer(result["model"], current_features, selected_customer, top_n=len(FEATURE_COLUMNS))
local_plot = local.sort_values("contribution_log_odds", ascending=True).copy()
local_plot["effect"] = np.where(
    local_plot["contribution_log_odds"].ge(0),
    "Pushes toward no purchase",
    "Pushes away from no purchase",
)
fig = px.bar(
    local_plot,
    x="contribution_log_odds",
    y="metric",
    color="effect",
    orientation="h",
    color_discrete_map={
        "Pushes toward no purchase": "#C2413B",
        "Pushes away from no purchase": "#287D70",
    },
    labels={
        "contribution_log_odds": "Contribution to model log-odds (relative to baseline)",
        "metric": "Metric",
        "effect": "Model effect",
    },
    title=f"Largest model contributions for customer {selected_customer}",
)
fig.add_vline(x=0, line_color="gray", line_width=1)
fig.update_layout(height=max(420, len(local_plot) * 28), showlegend=True)
st.plotly_chart(fig, use_container_width=True)

explanation_table = local[["metric", "value", "contribution_log_odds", "effect"]].copy()
explanation_table["value"] = explanation_table.apply(
    lambda r: f"${r['value']:,.2f}" if "spend" in str(r["metric"]).lower() or "value" in str(r["metric"]).lower() else f"{r['value']:,.2f}",
    axis=1,
)
st.dataframe(explanation_table, use_container_width=True, hide_index=True)
st.caption(
    "Positive contributions push this model's score toward 'no purchase'; negative contributions push it away. "
    "Contributions are additive in log-odds after feature scaling. They explain this fitted Logistic Regression, "
    "not causation, and correlated metrics can share or offset contributions."
)

st.subheader("Which metrics matter most overall?")
global_effects = global_feature_effects(result["model"])
top_global = global_effects.head(12).sort_values("coefficient_per_std")
top_global["effect"] = np.where(
    top_global["coefficient_per_std"].ge(0),
    "Higher values associate with more inactivity",
    "Higher values associate with less inactivity",
)
global_fig = px.bar(
    top_global,
    x="coefficient_per_std",
    y="metric",
    color="effect",
    orientation="h",
    color_discrete_map={
        "Higher values associate with more inactivity": "#C2413B",
        "Higher values associate with less inactivity": "#287D70",
    },
    labels={"coefficient_per_std": "Log-odds change per 1 SD increase", "metric": "Metric"},
    title="Largest standardized Logistic Regression coefficients",
)
global_fig.add_vline(x=0, line_color="gray", line_width=1)
global_fig.update_layout(height=450)
st.plotly_chart(global_fig, use_container_width=True)
st.dataframe(
    global_effects[["metric", "coefficient_per_std", "odds_ratio_per_std", "direction"]].head(15),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Global coefficients are conditional associations in this model. A positive coefficient means that a one-standard-deviation "
    "increase in that metric is associated with higher no-purchase odds, holding the other model inputs fixed. "
    "This is not a causal explanation or proof that changing the metric would change future behavior."
)
