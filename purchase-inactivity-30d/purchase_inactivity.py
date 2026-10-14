"""Lightweight, time-aware model for 30-day purchase inactivity.

Positive target (1): the customer purchased in the 30 days before a snapshot,
but made no purchase in the next 30 calendar days.

This module intentionally keeps the estimator small and explainable:
median imputation -> standard scaling -> regularized logistic regression.
It expects transaction-line data with customer, transaction, date, price, quantity.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = {
    "customer_id", "transaction_id", "transaction_date", "price", "quantity"
}
TARGET_COLUMN = "no_purchase_next_30d"

FEATURE_LABELS = {
    "orders_30d": "Orders in last 30 days",
    "spend_30d": "Spend in last 30 days",
    "units_30d": "Units bought in last 30 days",
    "active_days_30d": "Active purchase days in last 30 days",
    "orders_prev30d": "Orders in previous 30-day window",
    "spend_prev30d": "Spend in previous 30-day window",
    "orders_90d": "Orders in last 90 days",
    "spend_90d": "Spend in last 90 days",
    "units_90d": "Units bought in last 90 days",
    "active_days_90d": "Active purchase days in last 90 days",
    "categories_90d": "Distinct categories in last 90 days",
    "orders_365d": "Orders in last 365 days",
    "spend_365d": "Spend in last 365 days",
    "lifetime_orders": "Historical order count before snapshot",
    "lifetime_spend": "Historical spend before snapshot",
    "days_since_last_purchase": "Days since last purchase",
    "tenure_days": "Days since first observed purchase",
    "mean_gap_days": "Mean gap between distinct purchase days",
    "avg_order_value_90d": "Average order value in last 90 days",
    "order_change_30d": "Change in order count vs previous 30 days",
    "spend_change_30d": "Change in spend vs previous 30 days",
}

FEATURE_COLUMNS = list(FEATURE_LABELS)


def _normalise_date(value: Any) -> pd.Timestamp:
    """Return a timezone-naive, normalized timestamp (midnight)."""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError("Could not interpret the supplied date.")
    return pd.Timestamp(parsed).tz_convert(None).normalize()


def prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare transaction lines without using future-derived metrics."""
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame.")
    missing = REQUIRED_COLUMNS.difference(transactions.columns)
    if missing:
        raise ValueError("Missing required transaction columns: " + ", ".join(sorted(missing)))

    data = transactions.copy()
    data = data.loc[data["customer_id"].notna() & data["transaction_id"].notna()].copy()
    data["customer_id"] = data["customer_id"].astype(str).str.strip()
    data["transaction_id"] = data["transaction_id"].astype(str).str.strip()
    data = data.loc[data["customer_id"].ne("") & data["transaction_id"].ne("")].copy()

    data["order_date"] = (
        pd.to_datetime(data["transaction_date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    numeric_ok = (
        np.isfinite(data["price"].to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(data["quantity"].to_numpy(dtype=float, na_value=np.nan))
    )
    data = data.loc[
        data["order_date"].notna()
        & numeric_ok
        & data["price"].ge(0)
        & data["quantity"].gt(0)
    ].copy()
    if data.empty:
        raise ValueError("No valid transaction lines remain after checking IDs, dates, prices, and quantities.")

    if "category" not in data.columns:
        data["category"] = "__UNKNOWN__"
    else:
        data["category"] = data["category"].fillna("__UNKNOWN__").astype(str).str.strip()
        data.loc[data["category"].eq(""), "category"] = "__UNKNOWN__"

    data["line_spend"] = data["price"] * data["quantity"]
    return data.sort_values(["order_date", "customer_id", "transaction_id"]).reset_index(drop=True)


def _aggregate_window(
    lines: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
    suffix: str,
    include_categories: bool = False,
) -> pd.DataFrame:
    mask = lines["order_date"].lt(end)
    if start is not None:
        mask &= lines["order_date"].ge(start)
    window = lines.loc[mask]
    columns = [f"orders_{suffix}", f"spend_{suffix}", f"units_{suffix}", f"active_days_{suffix}"]
    if include_categories:
        columns.append(f"categories_{suffix}")
    if window.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="customer_id"))

    grouped = window.groupby("customer_id", sort=False)
    result = pd.DataFrame(index=grouped.size().index)
    result.index.name = "customer_id"
    result[f"orders_{suffix}"] = grouped["transaction_id"].nunique()
    result[f"spend_{suffix}"] = grouped["line_spend"].sum()
    result[f"units_{suffix}"] = grouped["quantity"].sum()
    result[f"active_days_{suffix}"] = grouped["order_date"].nunique()
    if include_categories:
        result[f"categories_{suffix}"] = grouped["category"].nunique()
    return result[columns]


def build_features_at_snapshot(
    prepared_lines: pd.DataFrame, snapshot_date: Any
) -> pd.DataFrame:
    """Create features as known strictly before a snapshot.

    Returns one row for each customer with at least one purchase in the 30 calendar
    days immediately before snapshot_date. Do not pass raw data from after the
    snapshot; prepare_transactions() itself does not censor data by snapshot.
    """
    snapshot = _normalise_date(snapshot_date)
    history = prepared_lines.loc[prepared_lines["order_date"].lt(snapshot)].copy()
    if history.empty:
        return pd.DataFrame(columns=["customer_id", *FEATURE_COLUMNS])

    recent = history.loc[history["order_date"].ge(snapshot - pd.Timedelta(days=30))]
    active_ids = pd.Index(recent["customer_id"].drop_duplicates(), name="customer_id")
    if active_ids.empty:
        return pd.DataFrame(columns=["customer_id", *FEATURE_COLUMNS])

    features = pd.DataFrame(index=active_ids)
    windows = [
        (snapshot - pd.Timedelta(days=30), "30d", False),
        (snapshot - pd.Timedelta(days=60), "prev30d", False),
        (snapshot - pd.Timedelta(days=90), "90d", True),
        (snapshot - pd.Timedelta(days=365), "365d", False),
        (None, "lifetime", False),
    ]
    for start, suffix, include_categories in windows:
        part = _aggregate_window(history, start, snapshot, suffix, include_categories)
        if suffix == "lifetime":
            part = part.rename(columns={"orders_lifetime": "lifetime_orders", "spend_lifetime": "lifetime_spend"})
        features = features.join(part.reindex(active_ids).fillna(0), how="left")

    distinct_days = history[["customer_id", "order_date"]].drop_duplicates().sort_values(
        ["customer_id", "order_date"]
    )
    grouped_days = distinct_days.groupby("customer_id", sort=False)["order_date"]
    first_dates = grouped_days.min()
    last_dates = grouped_days.max()
    features["days_since_last_purchase"] = (snapshot - last_dates.reindex(active_ids)).dt.days
    features["tenure_days"] = (snapshot - first_dates.reindex(active_ids)).dt.days

    gaps = grouped_days.apply(lambda dates: dates.diff().dt.days.mean() if len(dates) > 1 else 0.0)
    features["mean_gap_days"] = gaps.reindex(active_ids).fillna(0.0)
    features["avg_order_value_90d"] = (
        features["spend_90d"] / features["orders_90d"].replace(0, np.nan)
    ).fillna(0.0)
    features["order_change_30d"] = features["orders_30d"] - features["orders_prev30d"]
    features["spend_change_30d"] = features["spend_30d"] - features["spend_prev30d"]

    features = features.reindex(columns=FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)
    features.insert(0, "customer_id", features.index.astype(str))
    return features.reset_index(drop=True)


def build_training_dataset(
    transactions: pd.DataFrame,
    data_complete_through: Any,
    snapshot_step_days: int = 30,
) -> pd.DataFrame:
    """Build historical supervised examples using non-overlapping 30-day snapshots.

    At snapshot t, only customers with a purchase in [t-30 days, t) are included.
    Target is 1 when no purchase exists in [t, t+30 days), and 0 otherwise.
    Only snapshots whose full 30-day target window is covered by the confirmed
    data-completeness date are included.
    """
    if snapshot_step_days < 30:
        raise ValueError("snapshot_step_days must be at least 30 to avoid overlapping target windows.")
    lines = prepare_transactions(transactions)
    complete_through = _normalise_date(data_complete_through)
    observed_max = lines["order_date"].max()
    observed_min = lines["order_date"].min()
    if complete_through < observed_max:
        raise ValueError("The completeness date is earlier than a transaction in the uploaded data.")

    first_snapshot = observed_min + pd.Timedelta(days=30)
    last_snapshot = complete_through - pd.Timedelta(days=29)
    if first_snapshot > last_snapshot:
        return pd.DataFrame(columns=["snapshot_date", "customer_id", *FEATURE_COLUMNS, TARGET_COLUMN])

    snapshots = pd.date_range(first_snapshot, last_snapshot, freq=f"{snapshot_step_days}D")
    examples: list[pd.DataFrame] = []
    for snapshot in snapshots:
        features = build_features_at_snapshot(lines, snapshot)
        if features.empty:
            continue
        future_end = snapshot + pd.Timedelta(days=30)
        future_buyers = set(
            lines.loc[
                lines["order_date"].ge(snapshot) & lines["order_date"].lt(future_end),
                "customer_id",
            ].unique()
        )
        features["snapshot_date"] = snapshot
        features[TARGET_COLUMN] = (~features["customer_id"].isin(future_buyers)).astype("int8")
        examples.append(features[["snapshot_date", "customer_id", *FEATURE_COLUMNS, TARGET_COLUMN]])

    if not examples:
        return pd.DataFrame(columns=["snapshot_date", "customer_id", *FEATURE_COLUMNS, TARGET_COLUMN])
    return pd.concat(examples, ignore_index=True).sort_values(
        ["snapshot_date", "customer_id"]
    ).reset_index(drop=True)


def make_model() -> Pipeline:
    """Return the lightweight baseline model pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def global_feature_effects(model: Pipeline) -> pd.DataFrame:
    """Rank standardized logistic coefficients; association, not causation."""
    coefficients = model.named_steps["classifier"].coef_[0]
    table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "metric": [FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
            "coefficient_per_std": coefficients,
        }
    )
    table["direction"] = np.where(
        table["coefficient_per_std"].gt(0),
        "Higher values associate with more inactivity",
        "Higher values associate with less inactivity",
    )
    table["odds_ratio_per_std"] = np.exp(np.clip(table["coefficient_per_std"], -20, 20))
    table["absolute_effect"] = table["coefficient_per_std"].abs()
    return table.sort_values("absolute_effect", ascending=False).reset_index(drop=True)


def explain_customer(
    model: Pipeline,
    current_features: pd.DataFrame,
    customer_id: Any,
    top_n: int = 10,
) -> pd.DataFrame:
    """Explain one Logistic Regression score with additive log-odds contributions.

    Contributions exactly sum to the model's decision function (plus intercept),
    after imputation and scaling. They describe the model, not causal drivers.
    """
    selected = current_features.loc[current_features["customer_id"].astype(str).eq(str(customer_id))]
    if selected.empty:
        raise ValueError(f"Customer {customer_id!s} is not in the current eligible customer set.")
    x = selected[FEATURE_COLUMNS]
    transformed = model[:-1].transform(x)
    coefficients = model.named_steps["classifier"].coef_[0]
    contributions = transformed[0] * coefficients
    detail = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "metric": [FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
            "value": x.iloc[0].to_numpy(dtype=float),
            "contribution_log_odds": contributions,
        }
    )
    detail["effect"] = np.where(
        detail["contribution_log_odds"].ge(0),
        "Pushes toward no purchase",
        "Pushes away from no purchase",
    )
    detail["absolute_contribution"] = detail["contribution_log_odds"].abs()
    return detail.sort_values("absolute_contribution", ascending=False).head(top_n).reset_index(drop=True)


def fit_evaluate_and_score(
    transactions: pd.DataFrame,
    data_complete_through: Any,
) -> dict[str, Any]:
    """Time-split evaluation, final fit on labeled history, and current scoring."""
    lines = prepare_transactions(transactions)
    complete_through = _normalise_date(data_complete_through)
    if complete_through < lines["order_date"].max():
        raise ValueError("The completeness date is earlier than a transaction in the uploaded data.")

    training_data = build_training_dataset(lines, complete_through)
    if training_data.empty:
        raise ValueError(
            "There are no eligible historical snapshots. More transaction history or a later confirmed completeness date is needed."
        )
    snapshot_dates = sorted(pd.to_datetime(training_data["snapshot_date"]).unique())
    if len(snapshot_dates) < 3:
        raise ValueError(
            f"Only {len(snapshot_dates)} historical snapshot window(s) can be labeled. "
            "At least 3 are required for a small time-based holdout; more history is preferable."
        )

    n_test_dates = max(1, int(np.ceil(len(snapshot_dates) * 0.2)))
    n_test_dates = min(n_test_dates, len(snapshot_dates) - 2)
    test_start = pd.Timestamp(snapshot_dates[-n_test_dates]).normalize()
    train_rows = training_data[training_data["snapshot_date"] < test_start]
    test_rows = training_data[training_data["snapshot_date"] >= test_start]
    if train_rows[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "The earlier training period contains only one target class. "
            "More history or a different data-completeness window is needed to train the classifier."
        )

    holdout_model = make_model()
    holdout_model.fit(train_rows[FEATURE_COLUMNS], train_rows[TARGET_COLUMN])
    test_probabilities = holdout_model.predict_proba(test_rows[FEATURE_COLUMNS])[:, 1]
    y_test = test_rows[TARGET_COLUMN].to_numpy(dtype=int)
    has_both_classes = len(np.unique(y_test)) == 2
    evaluation = {
        "n_snapshots": len(snapshot_dates),
        "train_snapshots": int(train_rows["snapshot_date"].nunique()),
        "test_snapshots": int(test_rows["snapshot_date"].nunique()),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_start": str(test_start.date()),
        "test_end": str(pd.Timestamp(test_rows["snapshot_date"].max()).date()),
        "test_no_purchase_rate": float(np.mean(y_test)) if len(y_test) else float("nan"),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)) if has_both_classes else None,
        "average_precision": float(average_precision_score(y_test, test_probabilities)) if has_both_classes else None,
        "brier_score": float(brier_score_loss(y_test, test_probabilities)) if len(y_test) else None,
        "holdout_has_both_classes": bool(has_both_classes),
    }

    final_model = make_model()
    final_model.fit(training_data[FEATURE_COLUMNS], training_data[TARGET_COLUMN])

    current_features = build_features_at_snapshot(lines, complete_through + pd.Timedelta(days=1))
    if current_features.empty:
        predictions = pd.DataFrame(columns=["customer_id", "no_purchase_probability"])
    else:
        probabilities = final_model.predict_proba(current_features[FEATURE_COLUMNS])[:, 1]
        predictions = current_features[["customer_id"]].copy()
        predictions["no_purchase_probability"] = probabilities
        predictions = predictions.sort_values("no_purchase_probability", ascending=False).reset_index(drop=True)

    return {
        "model": final_model,
        "holdout_model": holdout_model,
        "evaluation": evaluation,
        "training_data": training_data,
        "current_features": current_features,
        "predictions": predictions,
        "data_complete_through": complete_through,
        "feature_columns": FEATURE_COLUMNS.copy(),
    }
