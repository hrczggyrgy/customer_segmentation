# 30-Day Purchase Inactivity Model

## Goal and target definition

This branch adds an optional, standalone Streamlit app. It does not alter the
existing customer segmentation pipeline or the primary `app.py` dashboard.

The model estimates which customers who purchased during the last 30 days are
least likely to purchase during the following 30 days. For a snapshot date `t`:

- **Eligible customer:** at least one purchase in `[t - 30 days, t)`.
- **Positive target (`1`):** no purchase in `[t, t + 30 days)`.
- **Negative target (`0`):** at least one purchase in `[t, t + 30 days)`.

The score is a probability estimate from a baseline classifier, not a guarantee
that someone will or will not buy, and not a calibrated churn probability.

## Leakage controls and data completeness

Features use only transaction data dated before the snapshot. Historical labels
are created only when the full future 30-day window is covered by the confirmed
source-data completeness date. Snapshots are spaced 30 days apart so future
label windows do not overlap.

The app requires an explicit confirmation of the data-complete-through date.
A latest transaction date alone does not establish completeness: a quiet day may
have no sales, or the export may be delayed. If the selected completeness date is
wrong, both labels and reported evaluation can be misleading.

The evaluation holds out the latest snapshot dates rather than randomly splitting
customer-snapshot rows. The latest holdout is used only for evaluation; after that,
the displayed scoring model is retrained on all labeled historical snapshots.

## Features

Features are deliberately simple and computed from prior transaction lines:

- Order count, spending, purchased units, and active purchase days in recent windows.
- Distinct categories and behavior over 90 days.
- Longer-term order/spend history and customer tenure.
- Days since last purchase and mean gap between distinct purchase days.
- Changes in order count and spending relative to the preceding 30-day window.

Order counts use distinct transaction IDs rather than line-item counts. Mean gap is
based on distinct purchase days; a zero can mean there is not enough repeated-date
history, not that purchases occur with zero-day gaps.

## Model and interpretation

The baseline is a scikit-learn pipeline: median imputation, standard scaling, and
regularized Logistic Regression. This is fast for portfolio-sized retail data and
easier to explain than a complex ensemble.

For an individual customer, the local explanation shows per-feature contributions
to the logistic decision score (log-odds). Positive contributions push the model
toward the no-purchase class; negative contributions push away from it. The global
chart ranks standardized coefficients. These explanations describe the fitted
model, not causal drivers. Correlated metrics can share or offset contributions.

The app reports ROC-AUC and average precision when the holdout contains both
classes, along with the holdout prevalence and Brier score. Brier score alone does
not establish calibration; calibration has not been independently validated. With
few snapshot windows or an unusual holdout, treat performance metrics as
preliminary. More complete historical data improves the reliability of validation.

## Run

Install existing dependencies and start the standalone app:

```bash
pip install -r requirements.txt
streamlit run inactivity_app.py
```

Run focused tests:

```bash
python -m unittest discover -s tests -p 'test_purchase_inactivity.py' -v
```

The implementation uses dependencies already listed in `requirements.txt`.
