#generate_metrics.py

import warnings

import pandas as pd
import polars as pl

warnings.filterwarnings('ignore')

def generate_advanced_metrics(lazy_transactions: pl.LazyFrame) -> pl.DataFrame:
    """
    Engineers 50+ advanced retail metrics strictly from the transaction schema.
    """
    # ==========================================
    # 1. BASE SETUP & MARKET BENCHMARKS
    # ==========================================
    df = lazy_transactions.with_columns([
        (pl.col('quantity') * pl.col('price')).alias('line_total'),
        pl.col('transaction_date').dt.date().alias('tx_date'),
        pl.col('transaction_date').dt.weekday().alias('weekday'),
        pl.col('transaction_date').dt.hour().alias('hour')
    ])

    # Establish global reference date for Recency
    max_date = df.select(pl.col('tx_date').max()).collect().item()
    ref_date = max_date + pd.Timedelta(days=1)
    
    # Establish total departments for Penetration metric
    total_depts = df.select(pl.col('department').n_unique()).collect().item()

    # Calculate Global Category & SKU Price Benchmarks (for premium/discount affinity)
    cat_benchmarks = df.group_by('category').agg([
        pl.col('price').quantile(0.75).alias('cat_p75_price'),
        pl.col('price').quantile(0.25).alias('cat_p25_price')
    ])
    sku_benchmarks = df.group_by('product_id').agg(
        pl.col('price').max().alias('sku_max_price')
    )
    
    # Join benchmarks back to main lines
    df = df.join(cat_benchmarks, on='category', how='left') \
           .join(sku_benchmarks, on='product_id', how='left')

    # Flag items bought at premium vs discount
    df = df.with_columns([
        (pl.col('price') >= pl.col('cat_p75_price')).cast(pl.Int32).alias('is_premium_purchase'),
        (pl.col('price') <= pl.col('cat_p25_price')).cast(pl.Int32).alias('is_bargain_purchase'),
        (pl.col('price') < (pl.col('sku_max_price') * 0.80)).cast(pl.Int32).alias('is_discounted_sku')
    ])

    # ==========================================
    # 2. BASKET ARCHITECTURE (Receipt Level)
    # ==========================================
    baskets = df.group_by(['customer_id', 'transaction_id']).agg([
        pl.col('line_total').sum().alias('basket_spend'),
        pl.col('quantity').sum().alias('basket_units'),
        pl.col('product_id').n_unique().alias('basket_skus'),
        pl.col('category').n_unique().alias('basket_cats'),
        pl.col('department').n_unique().alias('basket_depts'),
        pl.col('tx_date').first().alias('basket_date')
    ])

    # Roll up basket metrics to the customer
    customer_baskets = baskets.group_by('customer_id').agg([
        pl.col('basket_spend').max().alias('Max_Basket_Value'),
        pl.col('basket_spend').min().alias('Min_Basket_Value'),
        pl.col('basket_spend').std().fill_null(0).alias('Basket_Value_Variance'),
        pl.col('basket_units').std().fill_null(0).alias('UPT_Variance'),
        
        # Propensities
        (pl.col('basket_units') == 1).sum().alias('count_single_item_baskets'),
        (pl.col('basket_depts') > 1).sum().alias('count_cross_dept_baskets'),
        
        # Trajectories (Comparing last 3 to first 3 baskets)
        # Note: Baskets are sorted by date naturally in typical logs, but we sort to be safe
        pl.col('basket_spend').sort_by('basket_date').tail(3).mean().alias('avg_spend_last_3'),
        pl.col('basket_spend').sort_by('basket_date').head(3).mean().alias('avg_spend_first_3')
    ])

    # ==========================================
    # 3. CATEGORY CONCENTRATION
    # ==========================================
    # Find the max spend in any single category per customer
    cat_spend = df.group_by(['customer_id', 'category']).agg(pl.col('line_total').sum().alias('cat_spend'))
    fav_cat = cat_spend.group_by('customer_id').agg([
        pl.col('cat_spend').max().alias('Max_Spend_Single_Category'),
        pl.col('category').sort_by('cat_spend', descending=True).first().alias('Primary_Category_Name')
    ])

    # ==========================================
    # 4. MASTER CUSTOMER AGGREGATION
    # ==========================================
    customer_lines = df.group_by('customer_id').agg([
        # --- VOLUME & VALUE ---
        pl.col('line_total').sum().alias('Total_Lifetime_Spend'),
        pl.col('quantity').sum().alias('Total_Lifetime_Units'),
        pl.col('transaction_id').n_unique().alias('Total_Lifetime_Transactions'),
        
        # --- CATALOG ENGAGEMENT ---
        pl.col('category').n_unique().alias('Category_Width'),
        pl.col('department').n_unique().alias('Department_Width'),
        pl.col('product_id').n_unique().alias('SKU_Breadth'),
        
        # --- PRICE TIERING ---
        pl.col('price').median().alias('Median_Item_Price'),
        pl.col('price').std().fill_null(0).alias('Price_Variability_Score'),
        pl.col('is_premium_purchase').sum().alias('premium_item_count'),
        pl.col('is_bargain_purchase').sum().alias('bargain_item_count'),
        pl.col('is_discounted_sku').sum().alias('discounted_item_count'),
        
        # --- TEMPORAL RHYTHM ---
        (pl.lit(ref_date) - pl.col('tx_date').max()).dt.total_days().alias('Recency_Days'),
        (pl.col('tx_date').max() - pl.col('tx_date').min()).dt.total_days().alias('Tenure_Days'),
        pl.col('tx_date').n_unique().alias('Active_Days'),
        
        # Day/Time Affinities
        (pl.col('weekday') >= 6).sum().alias('weekend_item_count'), # 6=Sat, 7=Sun
        pl.col('hour').mode().first().alias('Time_of_Day_Mode'),
        
        # Purchase velocity setup (Unique sorted dates)
        pl.col('tx_date').unique().sort().alias('unique_dates')
    ])

    # ==========================================
    # 5. DERIVED METRICS, RATIOS, & JOINS
    # ==========================================
    # Collect the master frames
    master_df = customer_lines.collect() \
        .join(customer_baskets.collect(), on='customer_id', how='left') \
        .join(fav_cat.collect(), on='customer_id', how='left')

    # Calculate final derived metrics
    final_metrics = master_df.with_columns([
        # Core Averages
        (pl.col('Total_Lifetime_Spend') / pl.col('Total_Lifetime_Transactions')).alias('Average_Order_Value_AOV'),
        (pl.col('Total_Lifetime_Spend') / pl.col('Total_Lifetime_Units')).alias('Average_Unit_Retail_AUR'),
        (pl.col('Total_Lifetime_Units') / pl.col('Total_Lifetime_Transactions')).alias('Average_Units_Per_Transaction_UPT'),
        
        # Catalog Ratios
        (pl.col('Department_Width') / pl.lit(total_depts)).alias('Department_Penetration_Pct'),
        (pl.col('Max_Spend_Single_Category') / pl.col('Total_Lifetime_Spend')).alias('Category_Concentration_Score'),
        (pl.col('Total_Lifetime_Units') / pl.col('SKU_Breadth')).alias('Quantity_to_SKU_Ratio'),
        
        # Propensities (%)
        (pl.col('count_single_item_baskets') / pl.col('Total_Lifetime_Transactions')).alias('Single_Item_Basket_Propensity'),
        (pl.col('count_cross_dept_baskets') / pl.col('Total_Lifetime_Transactions')).alias('Cross_Department_Propensity'),
        (pl.col('premium_item_count') / pl.col('Total_Lifetime_Units')).alias('Premium_Category_Propensity'),
        (pl.col('bargain_item_count') / pl.col('Total_Lifetime_Units')).alias('Bargain_Category_Propensity'),
        (pl.col('discounted_item_count') / pl.col('Total_Lifetime_Units')).alias('Inferred_Discount_Affinity'),
        (pl.col('weekend_item_count') / pl.col('Total_Lifetime_Units')).alias('Weekend_Purchase_Propensity'),
        
        # Temporal / Rhythm Math
        pl.when(pl.col('Tenure_Days') > 0)
          .then(pl.col('Active_Days') / pl.col('Tenure_Days'))
          .otherwise(pl.lit(1.0))
          .alias('Active_Days_Ratio'),
        
        # Average Inter-Purchase Time (AIPT)
        pl.when(pl.col('unique_dates').list.len() > 1)
          .then(pl.col('unique_dates').list.eval(pl.element().diff().dt.total_days()).list.drop_nulls().list.mean())
          .otherwise(pl.lit(0.0))
          .alias('Average_InterPurchase_Time_AIPT'),
          
        # First to Second Velocity
        pl.when(pl.col('unique_dates').list.len() > 1)
          .then((pl.col('unique_dates').list.get(1) - pl.col('unique_dates').list.get(0)).dt.total_days())
          .otherwise(pl.lit(0))
          .alias('First_to_Second_Velocity_Days'),
          
        # Spend Trajectory
        pl.when(pl.col('avg_spend_first_3') > 0)
          .then(pl.col('avg_spend_last_3') / pl.col('avg_spend_first_3'))
          .otherwise(pl.lit(1.0))
          .alias('Spend_Trajectory_Slope')
    ])

    # Calculate Churn Risk Index (requires AIPT to be computed first)
    final_metrics = final_metrics.with_columns(
        pl.when(pl.col('Average_InterPurchase_Time_AIPT') > 0)
          .then(pl.col('Recency_Days') / pl.col('Average_InterPurchase_Time_AIPT'))
          .otherwise(pl.lit(0.0))
          .alias('Churn_Risk_Index_CRI')
    )

    # Clean up intermediate/temp columns
    cols_to_drop = [
        'unique_dates', 'premium_item_count', 'bargain_item_count', 
        'discounted_item_count', 'weekend_item_count', 'count_single_item_baskets',
        'count_cross_dept_baskets', 'avg_spend_last_3', 'avg_spend_first_3',
        'Max_Spend_Single_Category'
    ]
    
    return final_metrics.drop(cols_to_drop)

# ==========================================
# Example Execution
# ==========================================
if __name__ == "__main__":
    # Assuming `clean_transactions` is your validated DataFrame from earlier
    # Ensure it's lazy before passing it in
    # metrics_df = generate_advanced_metrics(clean_transactions.lazy())
    # print(metrics_df.head())
    pass