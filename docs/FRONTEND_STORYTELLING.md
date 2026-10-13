# Integrated frontend improvements

Run `streamlit run app.py` as before. Upload cleaned transactions and generate
segments. The existing five-tab dashboard now includes:

- Dynamic segment descriptors with cluster IDs, consistent Safe-palette colors,
  and hypotheses for marketing experiments, not validated action recommendations.
- Aligned customer-share and observed-spend-share bars above the tabs; filtering
  never changes the full-base denominators or profile benchmarks.
- Ordered profile heatmaps with sample sizes, fixed -3/+3 color limits, raw
  hover values, and an explicit warning that above average is not always better.
- Point/box/lasso customer selection and ID-based drill-down in Risk–Value.
  Default targets use repeat-buyer delay >=2 and a full-base repeat-buyer spend
  percentile. Insufficient-history customers are crosses at x=0 for display only.
- Existing distribution, normalized radar, PCA, and on-demand retention views.

The core feature engineering and modeling in segmentation.py are unchanged.
Pandas is used only on the customer-level presentation dataset, not to replace
the transaction pipeline. app.py's UI/control code is refactored; the invalid
PCA except clause and the incorrect force-regeneration cache clear are fixed.
Retention uses the full transaction base and is explicitly labeled as not
responding to segment filters. The final observed month can be incomplete.

Streamlit >=1.35 supports chart selection. An older-version fallback remains
in the rendering helper. USD formatting follows the original app convention.
New segment descriptions use full-base customer spend quartiles and repeat-buyer
median delay; fewer than 50% eligible repeat buyers means limited repeat history.

Tests: `python -m unittest discover -s tests -p 'test_frontend_storytelling.py' -v`

Before deployment, run a browser smoke test with actual transaction data:
upload and generate, filter segments, select points across multiple traces,
verify IDs in the drill-down, inspect PCA metadata, generate retention, then
upload different transactions and confirm that stale views cannot be reused.
