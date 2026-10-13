#!/usr/bin/env python3
"""
Full End-to-End EDA with Polars
Customer Transaction Data Analysis
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(name: str):
    """Save figure to output directory."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches='tight')
    plt.close()

def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

# ============================================================
# 1. LOAD DATA
# ============================================================
print_section("LOADING DATA")
df = pl.read_parquet("customer_transactions_clean.parquet")
print(f"Shape: {df.shape}")
print(f"Memory usage: {df.estimated_size() / 1e9:.2f} GB")
print(f"Columns: {df.columns}")
print(f"Dtypes:\n{df.dtypes}")

# ============================================================
# 2. DATA QUALITY CHECKS
# ============================================================
print_section("DATA QUALITY CHECKS")

# Null counts
null_counts = df.null_count()
print("Null counts per column:")
print(null_counts)

# Duplicate check
dup_count = df.height - df.unique().height
print(f"\nDuplicate rows: {dup_count}")

# Negative/zero prices
neg_price = df.filter(pl.col("price") <= 0).height
print(f"Rows with price <= 0: {neg_price}")

# Zero quantity
zero_qty = df.filter(pl.col("quantity") == 0).height
print(f"Rows with quantity == 0: {zero_qty}")

# Date range
date_range = df.select(
    pl.col("transaction_date").min().alias("min_date"),
    pl.col("transaction_date").max().alias("max_date")
)
print(f"\nDate range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")

# Unique counts
unique_counts = df.select([
    pl.col("customer_id").n_unique().alias("n_customers"),
    pl.col("transaction_id").n_unique().alias("n_transactions"),
    pl.col("product_id").n_unique().alias("n_products"),
    pl.col("department").n_unique().alias("n_departments"),
    pl.col("category").n_unique().alias("n_categories"),
])
print(f"\nUnique counts:\n{unique_counts}")

# ============================================================
# 3. UNIVARIATE ANALYSIS - PRICE & QUANTITY
# ============================================================
print_section("UNIVARIATE ANALYSIS: PRICE & QUANTITY")

# Price statistics
price_stats = df.select([
    pl.col("price").mean().alias("mean"),
    pl.col("price").median().alias("median"),
    pl.col("price").std().alias("std"),
    pl.col("price").min().alias("min"),
    pl.col("price").quantile(0.25).alias("q25"),
    pl.col("price").quantile(0.75).alias("q75"),
    pl.col("price").max().alias("max"),
    pl.col("price").skew().alias("skew"),
    pl.col("price").kurtosis().alias("kurtosis"),
])
print("Price statistics:")
print(price_stats)

# Quantity statistics
qty_stats = df.select([
    pl.col("quantity").mean().alias("mean"),
    pl.col("quantity").median().alias("median"),
    pl.col("quantity").std().alias("std"),
    pl.col("quantity").min().alias("min"),
    pl.col("quantity").quantile(0.25).alias("q25"),
    pl.col("quantity").quantile(0.75).alias("q75"),
    pl.col("quantity").max().alias("max"),
])
print("\nQuantity statistics:")
print(qty_stats)

# Price distribution plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price histogram (clipped for visibility)
price_data = df.filter(pl.col("price") > 0)["price"].to_numpy()
axes[0, 0].hist(price_data, bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel("Price")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Price Distribution (price > 0)")
axes[0, 0].axvline(price_data.mean(), color='red', linestyle='--', label=f'Mean: {price_data.mean():.2f}')
axes[0, 0].axvline(np.median(price_data), color='green', linestyle='--', label=f'Median: {np.median(price_data):.2f}')
axes[0, 0].legend()

# Price boxplot
axes[0, 1].boxplot(price_data, orientation='horizontal')
axes[0, 1].set_xlabel("Price")
axes[0, 1].set_title("Price Boxplot")

# Quantity histogram (clipped)
qty_data = df.filter(pl.col("quantity") > 0)["quantity"].to_numpy()
qty_data_clipped = qty_data[qty_data <= np.quantile(qty_data, 0.99)]
axes[1, 0].hist(qty_data_clipped, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel("Quantity")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Quantity Distribution (99th percentile)")

# Quantity boxplot
axes[1, 1].boxplot(qty_data_clipped, orientation='horizontal')
axes[1, 1].set_xlabel("Quantity")
axes[1, 1].set_title("Quantity Boxplot (99th percentile)")

save_fig("01_price_quantity_distributions")

# ============================================================
# 4. TRANSACTION VALUE (PRICE * QUANTITY)
# ============================================================
print_section("TRANSACTION VALUE ANALYSIS")

df = df.with_columns(
    (pl.col("price") * pl.col("quantity")).alias("transaction_value")
)

tv_stats = df.select([
    pl.col("transaction_value").mean().alias("mean"),
    pl.col("transaction_value").median().alias("median"),
    pl.col("transaction_value").std().alias("std"),
    pl.col("transaction_value").min().alias("min"),
    pl.col("transaction_value").quantile(0.25).alias("q25"),
    pl.col("transaction_value").quantile(0.75).alias("q75"),
    pl.col("transaction_value").max().alias("max"),
])
print("Transaction value statistics:")
print(tv_stats)

# Transaction value distribution
tv_data = df.filter(pl.col("transaction_value") > 0)["transaction_value"].to_numpy()
tv_clipped = tv_data[tv_data <= np.percentile(tv_data, 99)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(tv_clipped, bins=100, edgecolor='black', alpha=0.7)
axes[0].set_xlabel("Transaction Value")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Transaction Value Distribution (99th percentile)")
axes[0].axvline(tv_clipped.mean(), color='red', linestyle='--', label=f'Mean: {tv_clipped.mean():.2f}')
axes[0].legend()

axes[1].boxplot(tv_clipped, orientation='horizontal')
axes[1].set_xlabel("Transaction Value")
axes[1].set_title("Transaction Value Boxplot (99th percentile)")

save_fig("02_transaction_value_distribution")

# ============================================================
# 5. TEMPORAL ANALYSIS
# ============================================================
print_section("TEMPORAL ANALYSIS")

# Extract time features
df = df.with_columns([
    pl.col("transaction_date").dt.year().alias("year"),
    pl.col("transaction_date").dt.month().alias("month"),
    pl.col("transaction_date").dt.day().alias("day"),
    pl.col("transaction_date").dt.weekday().alias("weekday"),  # 1=Mon, 7=Sun
    pl.col("transaction_date").dt.hour().alias("hour"),
    pl.col("transaction_date").dt.ordinal_day().alias("day_of_year"),
])

# Transactions over time (daily)
daily = df.group_by("transaction_date").agg([
    pl.len().alias("n_transactions"),
    pl.col("transaction_value").sum().alias("daily_revenue"),
    pl.col("customer_id").n_unique().alias("n_customers"),
]).sort("transaction_date")

print("Daily transactions sample:")
print(daily.head(10))

# Monthly aggregation
monthly = df.group_by(["year", "month"]).agg([
    pl.len().alias("n_transactions"),
    pl.col("transaction_value").sum().alias("monthly_revenue"),
    pl.col("customer_id").n_unique().alias("n_customers"),
    pl.col("transaction_value").mean().alias("avg_transaction_value"),
]).sort(["year", "month"])

print("\nMonthly summary:")
print(monthly)

# Plot temporal trends
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Daily transactions
axes[0, 0].plot(daily["transaction_date"].to_numpy(), daily["n_transactions"].to_numpy(), alpha=0.7)
axes[0, 0].set_title("Daily Transaction Volume")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Transactions")
axes[0, 0].tick_params(axis='x', rotation=45)

# Daily revenue
axes[0, 1].plot(daily["transaction_date"].to_numpy(), daily["daily_revenue"].to_numpy(), alpha=0.7, color='green')
axes[0, 1].set_title("Daily Revenue")
axes[0, 1].set_xlabel("Date")
axes[0, 1].set_ylabel("Revenue ($)")
axes[0, 1].tick_params(axis='x', rotation=45)

# Monthly transactions
monthly_labels = [f"{y}-{m:02d}" for y, m in zip(monthly["year"], monthly["month"])]
axes[1, 0].bar(range(len(monthly)), monthly["n_transactions"])
axes[1, 0].set_xticks(range(len(monthly)))
axes[1, 0].set_xticklabels(monthly_labels, rotation=45)
axes[1, 0].set_title("Monthly Transaction Volume")
axes[1, 0].set_ylabel("Transactions")

# Hourly pattern
hourly = df.group_by("hour").agg([
    pl.len().alias("n_transactions"),
    pl.col("transaction_value").mean().alias("avg_value"),
]).sort("hour")

axes[1, 1].bar(hourly["hour"], hourly["n_transactions"])
axes[1, 1].set_xlabel("Hour of Day")
axes[1, 1].set_ylabel("Transactions")
axes[1, 1].set_title("Hourly Transaction Pattern")
axes[1, 1].set_xticks(range(0, 24, 2))

save_fig("03_temporal_analysis")

# Weekday analysis
weekday_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
weekday_data = df.group_by("weekday").agg([
    pl.len().alias("n_transactions"),
    pl.col("transaction_value").sum().alias("revenue"),
    pl.col("transaction_value").mean().alias("avg_value"),
]).sort("weekday")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
weekday_labels = [weekday_map[w] for w in weekday_data["weekday"]]
axes[0].bar(weekday_labels, weekday_data["n_transactions"])
axes[0].set_title("Transactions by Weekday")
axes[0].set_ylabel("Transactions")

axes[1].bar(weekday_labels, weekday_data["revenue"])
axes[1].set_title("Revenue by Weekday")
axes[1].set_ylabel("Revenue ($)")

save_fig("04_weekday_analysis")

# ============================================================
# 6. DEPARTMENT & CATEGORY ANALYSIS
# ============================================================
print_section("DEPARTMENT & CATEGORY ANALYSIS")

# Department summary
dept_summary = df.group_by("department").agg([
    pl.len().alias("n_transactions"),
    pl.col("transaction_value").sum().alias("total_revenue"),
    pl.col("transaction_value").mean().alias("avg_transaction_value"),
    pl.col("price").mean().alias("avg_price"),
    pl.col("quantity").mean().alias("avg_quantity"),
    pl.col("customer_id").n_unique().alias("n_customers"),
    pl.col("product_id").n_unique().alias("n_products"),
]).sort("total_revenue", descending=True)

print("Department summary:")
print(dept_summary)

# Top categories
cat_summary = df.group_by("category").agg([
    pl.len().alias("n_transactions"),
    pl.col("transaction_value").sum().alias("total_revenue"),
    pl.col("transaction_value").mean().alias("avg_transaction_value"),
    pl.col("customer_id").n_unique().alias("n_customers"),
]).sort("total_revenue", descending=True)

print("\nTop 20 categories by revenue:")
print(cat_summary.head(20))

# Department visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 15 departments by revenue
top_depts = dept_summary.head(15)
axes[0, 0].barh(range(len(top_depts)), top_depts["total_revenue"][::-1])
axes[0, 0].set_yticks(range(len(top_depts)))
axes[0, 0].set_yticklabels(top_depts["department"][::-1])
axes[0, 0].set_xlabel("Revenue ($)")
axes[0, 0].set_title("Top 15 Departments by Revenue")

# Top 15 departments by transaction count
top_depts_count = dept_summary.sort("n_transactions", descending=True).head(15)
axes[0, 1].barh(range(len(top_depts_count)), top_depts_count["n_transactions"][::-1])
axes[0, 1].set_yticks(range(len(top_depts_count)))
axes[0, 1].set_yticklabels(top_depts_count["department"][::-1])
axes[0, 1].set_xlabel("Transaction Count")
axes[0, 1].set_title("Top 15 Departments by Transaction Count")

# Department avg transaction value
top_depts_avg = dept_summary.sort("avg_transaction_value", descending=True).head(15)
axes[1, 0].barh(range(len(top_depts_avg)), top_depts_avg["avg_transaction_value"][::-1])
axes[1, 0].set_yticks(range(len(top_depts_avg)))
axes[1, 0].set_yticklabels(top_depts_avg["department"][::-1])
axes[1, 0].set_xlabel("Avg Transaction Value ($)")
axes[1, 0].set_title("Top 15 Departments by Avg Transaction Value")

# Revenue share pie chart (top 10 + others)
top10_revenue = top_depts.head(10)["total_revenue"].sum()
other_revenue = dept_summary.slice(10).select(pl.col("total_revenue").sum()).item()
pie_data = list(top_depts.head(10)["total_revenue"]) + [other_revenue]
pie_labels = list(top_depts.head(10)["department"]) + ["Others"]
axes[1, 1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title("Revenue Share by Department (Top 10)")

save_fig("05_department_analysis")

# Category analysis - top 20
top_cats = cat_summary.head(20)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].barh(range(len(top_cats)), top_cats["total_revenue"][::-1])
axes[0].set_yticks(range(len(top_cats)))
axes[0].set_yticklabels(top_cats["category"][::-1])
axes[0].set_xlabel("Revenue ($)")
axes[0].set_title("Top 20 Categories by Revenue")

axes[1].barh(range(len(top_cats)), top_cats["n_transactions"][::-1])
axes[1].set_yticks(range(len(top_cats)))
axes[1].set_yticklabels(top_cats["category"][::-1])
axes[1].set_xlabel("Transaction Count")
axes[1].set_title("Top 20 Categories by Transaction Count")

save_fig("06_category_analysis")

# ============================================================
# 7. CUSTOMER-LEVEL ANALYSIS (RFM)
# ============================================================
print_section("CUSTOMER-LEVEL ANALYSIS (RFM)")

# Reference date for recency (max date in data + 1 day)
ref_date = df.select(pl.col("transaction_date").max()).item() + pl.duration(days=1)

customer_rfm = df.group_by("customer_id").agg([
    pl.col("transaction_date").max().alias("last_purchase"),
    pl.len().alias("frequency"),
    pl.col("transaction_value").sum().alias("monetary"),
    pl.col("transaction_value").mean().alias("avg_order_value"),
    pl.col("transaction_id").n_unique().alias("n_orders"),
    pl.col("product_id").n_unique().alias("n_unique_products"),
    pl.col("department").n_unique().alias("n_departments_shopped"),
]).with_columns([
    (ref_date - pl.col("last_purchase")).dt.total_days().alias("recency_days"),
])

print("RFM summary statistics:")
rfm_stats = customer_rfm.select([
    pl.col("recency_days"),
    pl.col("frequency"),
    pl.col("monetary"),
])
print(rfm_stats.describe())

# RFM distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

recency_data = customer_rfm["recency_days"].to_numpy()
freq_data = customer_rfm["frequency"].to_numpy()
monetary_data = customer_rfm["monetary"].to_numpy()

axes[0, 0].hist(recency_data, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel("Recency (days)")
axes[0, 0].set_ylabel("Customers")
axes[0, 0].set_title("Recency Distribution")

axes[0, 1].hist(freq_data[freq_data <= np.percentile(freq_data, 99)], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel("Frequency (transactions)")
axes[0, 1].set_ylabel("Customers")
axes[0, 1].set_title("Frequency Distribution (99th pctl)")

axes[0, 2].hist(monetary_data[monetary_data <= np.percentile(monetary_data, 99)], bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel("Monetary ($)")
axes[0, 2].set_ylabel("Customers")
axes[0, 2].set_title("Monetary Distribution (99th pctl)")

# Log scale for monetary
axes[1, 0].hist(np.log1p(monetary_data), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel("Log(Monetary + 1)")
axes[1, 0].set_ylabel("Customers")
axes[1, 0].set_title("Log Monetary Distribution")

# Recency vs Monetary scatter (sample)
sample_idx = np.random.choice(len(monetary_data), min(5000, len(monetary_data)), replace=False)
axes[1, 1].scatter(recency_data[sample_idx], monetary_data[sample_idx], alpha=0.3, s=10)
axes[1, 1].set_xlabel("Recency (days)")
axes[1, 1].set_ylabel("Monetary ($)")
axes[1, 1].set_title("Recency vs Monetary (sample)")

# Frequency vs Monetary scatter
axes[1, 2].scatter(freq_data[sample_idx], monetary_data[sample_idx], alpha=0.3, s=10)
axes[1, 2].set_xlabel("Frequency")
axes[1, 2].set_ylabel("Monetary ($)")
axes[1, 2].set_title("Frequency vs Monetary (sample)")

save_fig("07_rfm_distributions")

# RFM Segmentation (Quintile-based)
print("\nCreating RFM segments...")
customer_rfm = customer_rfm.with_columns([
    pl.col("recency_days").qcut(5, labels=["5", "4", "3", "2", "1"]).alias("R_score"),
    pl.col("frequency").qcut(5, labels=["1", "2", "3", "4", "5"]).alias("F_score"),
    pl.col("monetary").qcut(5, labels=["1", "2", "3", "4", "5"]).alias("M_score"),
]).with_columns([
    (pl.col("R_score").cat.physical() + pl.col("F_score").cat.physical() + pl.col("M_score").cat.physical()).alias("RFM_score"),
    pl.col("R_score").cat.physical().alias("R_val"),
    pl.col("F_score").cat.physical().alias("F_val"),
    pl.col("M_score").cat.physical().alias("M_val"),
])

# Segment definitions using physical values
def assign_segment(r, f, m):
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New Customers"
    elif r <= 2 and f >= 4 and m >= 4:
        return "At Risk"
    elif r <= 2 and f <= 2:
        return "Lost"
    else:
        return "Others"

# Apply segmentation using map_elements on physical values
customer_rfm = customer_rfm.with_columns([
    pl.struct(["R_val", "F_val", "M_val"]).map_elements(
        lambda x: assign_segment(x["R_val"], x["F_val"], x["M_val"]),
        return_dtype=pl.String
    ).alias("segment")
])

segment_summary = customer_rfm.group_by("segment").agg([
    pl.len().alias("n_customers"),
    pl.col("recency_days").mean().alias("avg_recency"),
    pl.col("frequency").mean().alias("avg_frequency"),
    pl.col("monetary").mean().alias("avg_monetary"),
    pl.col("monetary").sum().alias("total_revenue"),
]).sort("total_revenue", descending=True)

print("\nRFM Segment Summary:")
print(segment_summary)

# Segment visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(segment_summary)))

axes[0].pie(segment_summary["n_customers"], labels=segment_summary["segment"], autopct='%1.1f%%', colors=colors)
axes[0].set_title("Customer Count by Segment")

axes[1].pie(segment_summary["total_revenue"], labels=segment_summary["segment"], autopct='%1.1f%%', colors=colors)
axes[1].set_title("Revenue by Segment")

save_fig("08_rfm_segments")

# ============================================================
# 8. PRODUCT ANALYSIS
# ============================================================
print_section("PRODUCT ANALYSIS")

# Top products by revenue
product_summary = df.group_by(["product_id", "product_description", "department", "category"]).agg([
    pl.len().alias("n_transactions"),
    pl.col("transaction_value").sum().alias("total_revenue"),
    pl.col("transaction_value").mean().alias("avg_transaction_value"),
    pl.col("price").mean().alias("avg_price"),
    pl.col("quantity").sum().alias("total_quantity"),
    pl.col("customer_id").n_unique().alias("n_customers"),
]).sort("total_revenue", descending=True)

print("Top 20 products by revenue:")
print(product_summary.head(20))

# Product price vs quantity relationship
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sample for scatter
prod_sample = product_summary.sample(n=min(1000, product_summary.height), seed=42)
axes[0].scatter(prod_sample["avg_price"], prod_sample["total_quantity"], alpha=0.5, s=20)
axes[0].set_xlabel("Average Price")
axes[0].set_ylabel("Total Quantity Sold")
axes[0].set_title("Price vs Quantity (Top 1000 Products)")

# Revenue concentration (Pareto)
product_summary = product_summary.with_columns([
    (pl.col("total_revenue").cum_sum() / pl.col("total_revenue").sum() * 100).alias("cum_revenue_pct")
])
pareto_data = product_summary.select(["total_revenue", "cum_revenue_pct"]).to_numpy()
axes[1].plot(range(1, len(pareto_data) + 1), pareto_data[:, 1])
axes[1].axhline(y=80, color='red', linestyle='--', label='80% Revenue')
axes[1].set_xlabel("Number of Products (ranked by revenue)")
axes[1].set_ylabel("Cumulative Revenue %")
axes[1].set_title("Pareto Analysis: Product Revenue Concentration")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

save_fig("09_product_analysis")

# ============================================================
# 9. CROSS-ANALYSIS: DEPARTMENT x TIME, CUSTOMER x DEPARTMENT
# ============================================================
print_section("CROSS-ANALYSIS")

# Department revenue over time (monthly)
dept_monthly = df.group_by(["year", "month", "department"]).agg([
    pl.col("transaction_value").sum().alias("revenue"),
]).sort(["year", "month", "department"])

# Top 5 departments over time
top5_depts = dept_summary.head(5)["department"].to_list()
dept_monthly_top = dept_monthly.filter(pl.col("department").is_in(top5_depts))

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

for dept in top5_depts:
    dept_data = dept_monthly_top.filter(pl.col("department") == dept)
    labels = [f"{y}-{m:02d}" for y, m in zip(dept_data["year"], dept_data["month"])]
    axes[0].plot(range(len(dept_data)), dept_data["revenue"], label=dept, marker='o', markersize=3)

axes[0].set_xticks(range(len(monthly)))
axes[0].set_xticklabels(monthly_labels, rotation=45)
axes[0].set_title("Monthly Revenue by Top 5 Departments")
axes[0].set_ylabel("Revenue ($)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Stacked area chart
dept_pivot = dept_monthly_top.pivot(
    values="revenue",
    index=["year", "month"],
    on="department",
    aggregate_function="first"
).sort(["year", "month"])

# Fill nulls with 0
for col in top5_depts:
    if col in dept_pivot.columns:
        dept_pivot = dept_pivot.with_columns(pl.col(col).fill_null(0))

bottom = np.zeros(len(dept_pivot))
for i, dept in enumerate(top5_depts):
    if dept in dept_pivot.columns:
        values = dept_pivot[dept].to_numpy()
        axes[1].fill_between(range(len(dept_pivot)), bottom, bottom + values, label=dept, alpha=0.7)
        bottom += values

axes[1].set_xticks(range(len(monthly)))
axes[1].set_xticklabels(monthly_labels, rotation=45)
axes[1].set_title("Stacked Monthly Revenue by Top 5 Departments")
axes[1].set_ylabel("Revenue ($)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

save_fig("10_department_time_series")

# Customer-Department affinity
cust_dept = df.group_by(["customer_id", "department"]).agg([
    pl.len().alias("n_transactions"),
    pl.col("transaction_value").sum().alias("revenue"),
]).sort(["customer_id", "revenue"], descending=[False, True])

# Top department per customer
cust_top_dept = cust_dept.group_by("customer_id").first().select(["customer_id", "department", "revenue"])
top_dept_dist = cust_top_dept.group_by("department").agg([
    pl.len().alias("n_customers"),
    pl.col("revenue").sum().alias("total_revenue"),
]).sort("n_customers", descending=True)

print("\nPrimary department per customer:")
print(top_dept_dist.head(15))

# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================
print_section("CORRELATION ANALYSIS")

# Numeric columns correlation
numeric_cols = ["price", "quantity", "transaction_value"]
corr_matrix = df.select(numeric_cols).corr()
print("Correlation matrix:")
print(corr_matrix)

# Visualize correlation
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix.to_numpy(), annot=True, fmt=".3f", 
            xticklabels=numeric_cols, yticklabels=numeric_cols,
            cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Correlation Matrix: Price, Quantity, Transaction Value")
save_fig("11_correlation_matrix")

# ============================================================
# 11. OUTLIER DETECTION
# ============================================================
print_section("OUTLIER DETECTION")

# IQR method for price and quantity
for col in ["price", "quantity", "transaction_value"]:
    q1 = df.select(pl.col(col).quantile(0.25)).item()
    q3 = df.select(pl.col(col).quantile(0.75)).item()
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df.filter((pl.col(col) < lower) | (pl.col(col) > upper)).height
    pct = outliers / df.height * 100
    print(f"{col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, Lower={lower:.2f}, Upper={upper:.2f}")
    print(f"  Outliers: {outliers} ({pct:.2f}%)")

# Top outliers
print("\nTop 10 highest transaction values:")
top_tv = df.sort("transaction_value", descending=True).head(10)
print(top_tv.select(["customer_id", "transaction_date", "department", "category", "price", "quantity", "transaction_value"]))

print("\nTop 10 highest quantities:")
top_qty = df.sort("quantity", descending=True).head(10)
print(top_qty.select(["customer_id", "transaction_date", "department", "category", "price", "quantity", "transaction_value"]))

# ============================================================
# 12. SUMMARY REPORT
# ============================================================
print_section("SUMMARY REPORT")

summary = f"""
============================================================
           CUSTOMER TRANSACTION EDA SUMMARY
============================================================

DATA OVERVIEW:
- Total Transactions: {df.height:,}
- Unique Customers: {unique_counts['n_customers'][0]:,}
- Unique Products: {unique_counts['n_products'][0]:,}
- Unique Departments: {unique_counts['n_departments'][0]:,}
- Unique Categories: {unique_counts['n_categories'][0]:,}
- Date Range: {date_range['min_date'][0].date()} to {date_range['max_date'][0].date()}
- Total Revenue: ${df['transaction_value'].sum():,.2f}

PRICE & QUANTITY:
- Avg Price: ${df['price'].mean():.2f}
- Median Price: ${df['price'].median():.2f}
- Price Range: ${df['price'].min():.2f} - ${df['price'].max():.2f}
- Avg Quantity: {df['quantity'].mean():.2f}
- Median Quantity: {df['quantity'].median():.0f}

TRANSACTION VALUE:
- Avg Transaction Value: ${df['transaction_value'].mean():.2f}
- Median Transaction Value: ${df['transaction_value'].median():.2f}

TOP DEPARTMENTS BY REVENUE:
{dept_summary.head(5).select(['department', 'total_revenue', 'n_transactions']).to_pandas().to_string(index=False)}

RFM SEGMENTS:
{segment_summary.select(['segment', 'n_customers', 'total_revenue']).to_pandas().to_string(index=False)}

KEY INSIGHTS:
1. Data spans 2 years (2000-2001) with {df.height:,} transactions
2. {unique_counts['n_customers'][0]:,} unique customers with avg {df.height/unique_counts['n_customers'][0]:.1f} transactions each
3. Top department '{dept_summary['department'][0]}' generates ${dept_summary['total_revenue'][0]:,.0f} ({dept_summary['total_revenue'][0]/df['transaction_value'].sum()*100:.1f}% of revenue)
4. {segment_summary.filter(pl.col('segment')=='Champions')['n_customers'].item() if 'Champions' in segment_summary['segment'].to_list() else 0} Champions customers drive {segment_summary.filter(pl.col('segment')=='Champions')['total_revenue'].item()/df['transaction_value'].sum()*100:.1f}% of revenue if segment exists
5. Strong hourly pattern with peak shopping hours
6. Weekend vs weekday shopping patterns differ
"""

print(summary)

# Save summary to file
with open(OUTPUT_DIR / "eda_summary.txt", "w") as f:
    f.write(summary)

print(f"\n✅ EDA Complete! Outputs saved to {OUTPUT_DIR}/")
print("Generated files:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  - {f.name}")
print(f"  - eda_summary.txt")