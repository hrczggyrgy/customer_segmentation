# Customer Segmentation Platform

A production-ready customer segmentation pipeline built with **Polars**, **scikit-learn**, and **Streamlit**. Processes 2.5M+ retail transactions to identify actionable customer segments using RFM analysis and K-Means clustering.

## Architecture

```
├── app.py                    # Streamlit dashboard (Risk vs Value Action Center)
├── segmentation.py           # Advanced segmentation pipeline (PCA + KMeans + Silhouette)
├── visual.py                 # Cluster visualization suite (radar, heatmap, violin plots)
├── base_scripts/
│   ├── generate_metrics.py   # 50+ retail feature engineering metrics
│   ├── customer_segmentation.py  # Core RFM + clustering pipeline
│   └── eda_polars.py         # Full EDA with 12 analytical sections
├── requirements.txt          # Dependencies
├── customer_segments.parquet # Generated segment assignments (2,500 customers)
├── customer_transactions_clean.parquet  # Cleaned transaction data (2.5M rows)
└── eda_outputs/              # 12 EDA visualizations + summary report
```

## Key Features

| Component | Description |
|-----------|-------------|
| **Data Processing** | Polars lazy execution for 2.5M+ rows with schema validation |
| **Feature Engineering** | 50+ metrics: RFM, tenure, category diversity, inter-purchase timing, price sensitivity |
| **Segmentation** | PCA dimensionality reduction → KMeans with silhouette optimization |
| **Segments Identified** | Champions, Loyal, New, At-Risk, Lost, Others |
| **Dashboard** | Streamlit app with Risk vs Value matrix, CRM export, drill-down views |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/hrczggyrgy/customer_segmentation.git
cd customer_segmentation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run segmentation pipeline
python segmentation.py

# Launch dashboard
streamlit run app.py
```

## Segmentation Results (Clean Data)

| Cluster | Label | Customers | Avg Recency | Avg Frequency | Avg Spend | Avg Order Value |
|---------|-------|-----------|-------------|---------------|-----------|-----------------|
| 0 | At-Risk / Lapsed | 791 (31.6%) | 64.5 days | 28 | $643 | $26.92 |
| 1 | Active / Loyal | 1,709 (68.4%) | 10.3 days | 134 | $4,007 | $34.59 |

**Cluster 1 drives 85% of revenue** — prime targets for loyalty program. **Cluster 0** needs reactivation campaigns.

## Data Pipeline

1. **Clean**: Remove gas station (KIOSK-GAS) and coupon/misc items (37K outlier rows)
2. **Validate**: Schema enforcement, null handling, type casting via Polars expressions
3. **Feature Engineering**: RFM + advanced metrics (50+ features per customer)
4. **Scale**: Log1p → StandardScaler → PCA (6 components)
5. **Cluster**: KMeans with silhouette score optimization (K=2 optimal)
6. **Visualize**: Radar charts, heatmaps, violin plots, risk-value matrix
7. **Deploy**: Streamlit dashboard for business users

## EDA Highlights

- **2.5M transactions** across 2,500 customers, 92K products, 39 departments (2000-2001)
- **Top departments**: Grocery (55%), Drug GM (14%), Produce (7.6%), Meat (7.5%)
- **Outliers removed**: Gas bulk purchases (89K qty @ $0.002), zero-price/quantity rows
- **RFM distributions**: Highly right-skewed → log-transformed for clustering

## Requirements

- Python 3.10+
- Polars 1.0+, Pandas 2.0+, NumPy 1.24+
- scikit-learn 1.3+
- Matplotlib, Seaborn, Plotly, Streamlit

## License

MIT License — feel free to use and adapt for your retail analytics needs.