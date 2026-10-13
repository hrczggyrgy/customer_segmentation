# Customer Segmentation Platform

A production-ready customer segmentation pipeline built with **Polars**, **scikit-learn**, and **Streamlit**. Processes 2.5M+ retail transactions to identify actionable customer segments using RFM analysis and K-Means clustering.

## Architecture

```
├── app.py                    # Streamlit dashboard (Risk vs Value, Profiles, Distributions, PCA, Retention)
├── segmentation.py           # Advanced segmentation pipeline (Winsorization + RobustScaler + PCA + KMeans)
├── visual.py                 # Cluster visualization suite (radar, heatmap, violin plots)
├── base_scripts/
│   ├── generate_metrics.py   # 50+ retail feature engineering metrics
│   ├── customer_segmentation.py  # Core RFM + clustering pipeline
│   └── eda_polars.py         # Full EDA with 12 analytical sections
├── requirements.txt          # Dependencies
├── pca_metadata.json         # PCA explained variance & loadings (auto-generated)
├── customer_segments_advanced.parquet  # Generated segment assignments (2,500 customers)
├── customer_transactions_clean.parquet  # Cleaned transaction data (2.5M rows)
└── eda_outputs/              # 12 EDA visualizations + summary report
```

## Key Features

| Component | Description |
|-----------|-------------|
| **Data Processing** | Polars lazy execution for 2.5M+ rows with schema validation |
| **Feature Engineering** | 50+ metrics: RFM, tenure, category diversity, inter-purchase timing, price sensitivity |
| **Segmentation** | Winsorization (99th pctl) → RobustScaler (median/IQR) → PCA → KMeans with Davies-Bouldin optimization |
| **Segments Identified** | k=7 clusters (min k=3 enforced for actionable CRM) |
| **Dashboard** | Streamlit app: Risk–Value matrix, benchmark heatmap, distributions, PCA, retention |

## Segmentation Pipeline (Robust)

| Step | Method | Why |
|------|--------|-----|
| **Outlier Capping** | Winsorization at 99th percentile | Caps heavy tails that dominate variance |
| **Scaling** | RobustScaler (median/IQR) | Immune to outlier pull vs StandardScaler |
| **Dimensionality** | PCA (6 components, 85% variance) | Denoises before clustering |
| **Cluster Selection** | Davies-Bouldin Index (k≥3) | Avoids Silhouette's bias toward k=2 |
| **Min Clusters** | k=3 enforced | Ensures actionable CRM segmentation |

## Segmentation Results (Clean Data)

| Cluster | Size | Avg Spend | Delay Index | Persona |
|---------|------|-----------|-------------|---------|
| 0 | 584 | $5,842 | 0.93 | High-value loyal |
| 1 | 68 | $2,156 | 3.21 | **At-risk whales** |
| 2 | 156 | $1,842 | 1.45 | Fading mid-value |
| 3 | 482 | $4,921 | 1.02 | **Champions** |
| 4 | 719 | $1,423 | 0.87 | Regular shoppers |
| 5 | 52 | $234 | 4.12 | Low-value at-risk |
| 6 | 239 | $567 | 0.81 | Occasional buyers |

**Cluster 3 (Champions) + Cluster 0 (Loyal)** drive ~70% of revenue. **Cluster 1** is the primary reactivation target.

## Dashboard Tabs

| Tab | Purpose |
|-----|---------|
| **Risk–Value Portfolio** | Purchase-Delay Index vs Spend with quadrant annotations, high-value threshold |
| **Segment Profiles** | Z-score heatmap vs overall customer base (red=above, blue=below), radar as expander |
| **Distributions** | Selectable box plots with log toggle + summary stats |
| **Model Exploration** | 2D/3D PCA projections with actual explained variance |
| **Retention** | Monthly repeat-purchase cohort matrix (on-demand generation) |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/hrczggyrgy/customer_segmentation.git
cd customer_segmentation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run segmentation pipeline (generates segments + PCA metadata)
python3 segmentation.py

# Launch dashboard
streamlit run app.py
# Opens at http://localhost:8501
```

## Streamlit Cloud Deployment

1. Push to GitHub (already configured)
2. Go to https://share.streamlit.io → New app → Select repo
3. Main file: `app.py`
4. Deploy!
5. On cloud: Upload `customer_transactions_clean.parquet` → Click **"Generate / Refresh Segments"**

## Data Pipeline

1. **Clean**: Remove gas station (KIOSK-GAS) and coupon/misc items (37K outlier rows removed)
2. **Validate**: Schema enforcement, null handling, type casting via Polars expressions
3. **Feature Engineering**: RFM + advanced metrics (50+ features per customer)
4. **Robust Preprocessing**: Winsorization → Log1p → RobustScaler (median/IQR) → PCA
5. **Cluster**: KMeans with Davies-Bouldin optimization (k=7 optimal)
6. **Visualize**: Risk-value matrix, benchmark heatmap, distributions, PCA, retention
7. **Deploy**: Streamlit dashboard for business users

## Requirements

- Python 3.10+
- Polars 1.0+, Pandas 2.0+, NumPy 1.24+
- scikit-learn 1.3+
- Matplotlib, Seaborn, Plotly, Streamlit

## License

MIT License — feel free to use and adapt for your retail analytics needs.