# Customer Segmentation Platform

A customer-analytics portfolio project using Polars, scikit-learn, Streamlit,
and Plotly. The app connects behavioral segmentation to business-oriented
comparisons and customer-level exploration. It is not a production-security
certification, and full browser validation with the actual datasets remains required.

## Architecture

```text
app.py                         Streamlit dashboard and session lifecycle
frontend_storytelling.py       Segment descriptions, contribution charts, profiles, drill-down
data_safety.py                 Shared SHA-256, session-only results, finalized retention
segmentation.py                Feature engineering, robust preprocessing, PCA, KMeans
visual.py                      Standalone visualization suite
base_scripts/                  Original EDA and feature-engineering scripts
tests/test_frontend_storytelling.py  Presentation helper tests
tests/test_data_safety.py       Fingerprint, isolation, completeness, retention tests
docs/FRONTEND_STORYTELLING.md   Frontend interpretation and validation notes
```

## Dashboard

- Customer share versus observed-spend share with stable full-base denominators.
- Behavior-derived segment descriptions retaining cluster IDs and consistent colors.
- Benchmark heatmaps with sample sizes, raw-value hover, and fixed color bounds.
- Risk-value scatterplots with point/box/lasso selection and customer-ID drill-down.
- Distribution plots, optional normalized radar profiles, PCA projections,
  actual explained variance and signed component loadings.
- On-demand monthly repeat-purchase retention with a visible finalized cutoff.

Historical spend is not predicted lifetime value. Purchase delay is not a
calibrated churn probability. Marketing actions are hypotheses for experiments.
Cluster IDs, cluster counts, and descriptive labels may change with the dataset.
Previous example results are dataset-specific, not guaranteed outcomes.

## Data identity and deployment behavior

Upload, local-file, and standalone workflows call the same full-length SHA-256
helper on source file bytes. Different Parquet encodings can have different
fingerprints even if their logical rows match. Legacy 16-character fingerprints
are not silently treated as equivalent; regenerate old artifacts when needed.

The dashboard keeps transaction data, segment assignments, PCA metadata, and
retention results in session state. It does not read or write generated segment
or metadata files and does not place these results in a shared Streamlit cache.
Refreshing reuses only matching results within that session; force regeneration
recomputes only that session. Disconnecting/reloading can lose session results.

`python segmentation.py` remains a controlled offline workflow that writes
`customer_segments_advanced.parquet` and `pca_metadata.json` with a full source
fingerprint. Those files are not consumed by the web dashboard. Do not run that
writer as a multi-user upload persistence service.

Session isolation does not supply authentication, encryption, a retention-policy
implementation, or a compliance guarantee. Use synthetic/anonymized demo data
for a public portfolio; do not advertise unrestricted sensitive-data uploads.

## Finalized retention

By default, the latest observed calendar month is excluded, even if the last
transaction falls on month-end. Earlier months are assumed complete. When source
completeness is known, enable the confirmed data-complete-through date. Only
whole calendar months ending on or before that date are finalized.

Unfinished periods are null; observed periods with no returning customers are
0%. Cohorts after the cutoff are excluded. Missing observed period columns are
created as zero rather than silently omitted. Month 1/3 summaries average eligible
cohorts equally and exclude nulls. The last transaction date alone cannot prove
source completeness.

## Run

Python 3.10+ is required. Install dependencies, then launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload cleaned transaction Parquet and click Generate segments. Alternatively,
place `customer_transactions_clean.parquet` locally as a controlled demo source.
The segmentation algorithm remains Polars feature engineering, winsorization,
log transforms, robust scaling, PCA, and KMeans with Davies-Bouldin selection.

## Validate

```bash
python -m unittest discover -s tests -v
```

Polars integration tests run when Polars is installed; otherwise they are
explicitly skipped. Session tests use a fake pipeline and do not prove browser
or model runtime behavior. Before deployment, test two independent sessions,
upload/force/switch-dataset flows, chart selection IDs, PCA metadata, and partial
versus source-confirmed complete-month retention using the actual app.

## License

MIT License; retain the repository's licensing terms.
