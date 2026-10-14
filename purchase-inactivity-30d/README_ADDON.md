# Add-on files: 30-day purchase inactivity model

These files are designed to be copied into the root of `hrczggyrgy/customer_segmentation`
on a new branch named `purchase-inactivity-30d`. Existing application and model files
are not modified.

New files:
- `purchase_inactivity.py` — feature engineering, point-in-time labels, training,
  time-based evaluation, current scoring, and explanations.
- `inactivity_app.py` — standalone Streamlit dashboard.
- `tests/test_purchase_inactivity.py` — focused tests for label boundaries and leakage.
- `docs/PURCHASE_INACTIVITY_MODEL.md` — target, assumptions, limitations, and run guide.

From a local clone of the repository:

```bash
git checkout main
git pull
git checkout -b purchase-inactivity-30d
# Copy the files/folders from this add-on into the repository root.
python -m unittest discover -s tests -p 'test_purchase_inactivity.py' -v
streamlit run inactivity_app.py
```

The existing `requirements.txt` already contains pandas, NumPy, scikit-learn,
Plotly, Polars, and Streamlit. Do not merge the branch until the focused tests and
an end-to-end run with the real transaction file pass.
