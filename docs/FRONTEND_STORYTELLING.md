# Frontend interpretation and deployment checks

The existing five-tab dashboard includes segment contribution charts,
behavior-derived labels, benchmark heatmaps, distributions/radar, interactive
customer drill-down, PCA diagnostics, and on-demand retention.

`frontend_storytelling.py` handles customer-level presentation. `data_safety.py`
handles full-length fingerprints, session-only segmentation results, and the
finalized retention boundary. Modeling logic in AdvancedCustomerSegmentation
is unchanged.

Spend shares and heatmap benchmarks use the full customer base. Observed spend
is historical, delay is not churn probability, and campaigns are hypotheses.
Descriptions depend on the data; keep cluster IDs for traceability. Insufficient
repeat history uses crosses at x=0 as a plotting convention. Non-positive spend
is omitted only from logarithmic charts, not from summaries.

## Observed behavior, hypotheses, and predictions

- Observed behavior: spend, AOV, recency, transactions, categories, purchase
  intervals, discount affinity, retention percentages, and segment averages are
  descriptive summaries of the uploaded transactions.
- Hypotheses: segment labels and the action hypothesis column are prompts for
  experiments. They are not validated recommendations.
- Predictions: the dashboard makes none. Observed spend is not predicted lifetime
  value, purchase delay is not a calibrated churn probability, and standardized
  differences are not significance tests or evidence of causality.

## Session-only data handling and limits

Dashboard results are not persisted as shared files or placed in a global
Streamlit cache. Transactions, segments, PCA metadata, and retention results
(including cohort sizes) live in session state, are reused only when the source
fingerprint (and retention cutoff) match, and can be lost on reload or
disconnect. Session-only state is not a substitute for authentication,
transport security, compliance, or safe data handling. Prefer anonymized demo data.

## Contribution index and segment differentiators

The leading segment is the one with highest observed-spend share among the
selected segments; shares always retain full-base denominators. Its contribution
index is spend share divided by customer share. A 35% spend share and 20% customer
share give 1.75x. The share gap is 35 - 20 = +15 percentage points, not a time
change or relative percentage increase. Cards display N/A for a zero customer
share, non-positive total base spend, or invalid/non-finite input. Zero spend
share is valid when total base spend and customer share are positive.

Below the profile heatmap, each segment card shows up to three differentiators
ranked by absolute (segment mean - full-base mean) / full-base standard deviation.
Constant metrics, missing/non-finite means, zero differences, and differences
below 0.25 standard deviations are omitted. The 0.25 threshold controls display
noise; it is not a statistical significance threshold. Ties follow metric order.
Changing selected segments does not recalculate the full-base benchmark.

Each trait shows direction (above/below), the standardized difference, the raw
segment and full-base averages with units, a feature-specific interpretation,
and a short definition taken from `segmentation.py`:

| Metric | Definition used | Unit shown |
| --- | --- | --- |
| Observed spend | Sum of quantity x price over observed lines | $ |
| AOV | Observed spend / distinct transaction IDs | $ |
| Recency | Days from last observed purchase to the day after the latest data date; higher = longer ago | days |
| Transactions | Distinct transaction IDs | count |
| Categories | Distinct product categories | count |
| Purchase interval | Mean gap between distinct purchase dates; 0 with fewer than two dates | days |
| Purchase delay | Recency / purchase interval when interval > 0, else 0 | ratio (x) |
| Discount affinity | Lines priced below 80% of the SKU maximum observed price / purchased units | flagged lines per unit |

Discount affinity is not a discount percentage or a share of transactions. A zero
purchase delay or interval comes from insufficient repeat history and does not
mean healthy behavior. Correlated traits are not independent evidence, and
summaries do not establish causality or better performance.

## Retention cohort sizes and unfinished periods

Each retention row label shows the cohort's starting customer count, for example
`2025-01 (n=420)`. Counts are distinct customer IDs by first observed purchase
month, computed from the full transaction base independently of the segment
filters and cached in session state with the retention matrix. Hover shows cohort,
months since first observed purchase, retention percentage, and retained customers
(percentage x cohort size, so the two stay aligned).

Retention excludes the latest observed month by default. An explicitly confirmed
data-complete-through date can include a known complete month. The cutoff is
visible; unfinished periods are null (blank, no hover) and finalized absences
remain 0%. Earlier source months must actually be complete for these assumptions
to hold. `data_safety.retention_matrix()` is unchanged.

## PCA labels and segment centers

PC axis labels include explained-variance percentages only when PCA metadata is
valid and its source fingerprint matches the loaded transactions; otherwise plain
PC names are shown with a note. Variance values are never hardcoded. Diamond
markers labeled C# show each displayed segment's center, the mean PC coordinates
of its displayed customers, in the 2D and 3D projections. A low-dimensional
projection does not establish separation in the full clustering space.

## Distributions selector

The Distributions tab offers Boxplot (default) or Violin plot (with an inner box)
for the selected metric. Segment filters, colors, category order, and the
logarithmic axis option apply to both. Non-positive observations omitted from a
logarithmic chart are counted in a caption; the summary-statistics table still
uses all selected data. Empty or non-plottable data shows a message.
