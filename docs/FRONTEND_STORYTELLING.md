# Frontend interpretation and deployment checks

The existing five-tab dashboard includes segment contribution charts,
behavior-derived labels, benchmark heatmaps, distributions/radar, interactive
customer drill-down, PCA diagnostics, and on-demand retention.

`frontend_storytelling.py` handles customer-level presentation. `data_safety.py`
handles full-length fingerprints, session-only segmentation results, and the
finalized retention boundary. Modeling logic in AdvancedCustomerSegmentation
is unchanged; only the standalone fingerprint caller changes.

Spend shares and heatmap benchmarks use the full customer base. Observed spend
is historical, delay is not churn probability, and campaigns are hypotheses.
Descriptions depend on the data; keep cluster IDs for traceability. Insufficient
repeat history uses crosses at x=0 as a plotting convention. Non-positive spend
is omitted only from logarithmic charts, not from summaries.

Dashboard results are not persisted as shared files or placed in a global
Streamlit cache. Session-only state is not a substitute for authentication,
transport security, compliance, or safe data handling. Prefer anonymized demo data.

Retention excludes the latest observed month by default. An explicitly confirmed
data-complete-through date can include a known complete month. The cutoff is
visible; immature periods are null and finalized absences remain 0%. Earlier
source months must actually be complete for these assumptions to hold.

Run `python -m unittest discover -s tests -v`. Pure helper/session tests use
synthetic data; Polars integration tests are skipped if Polars is unavailable.
A real browser smoke test remains required: upload, force regenerate, switch
datasets, use independent sessions, verify selected IDs and PCA metadata, and
compare finalized retention with source-confirmed complete months.
