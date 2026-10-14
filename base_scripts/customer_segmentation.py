#customer_segmentation.py               

import warnings

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class PolarsCustomerSegmentation:
    def __init__(self, max_clusters=10, random_state=42):
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.optimal_k = None
        self.model = None
        self.reference_date = None

    def validate_and_clean(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Enforces schema constraints using Polars expressions on a LazyFrame.
        """
        # 1. Clean strings (strip whitespace)
        str_cols = ['customer_id', 'transaction_id', 'product_id', 
                    'product_description', 'department', 'category']
        
        df_cleaned = lazy_df.with_columns([
            pl.col(c).cast(pl.Utf8).str.strip_chars() 
            for c in str_cols if c in lazy_df.collect_schema().names()
        ])

        # 2. Handle missing/null string replacements
        df_cleaned = df_cleaned.with_columns([
            # product_description -> empty string if null
            pl.col('product_description').fill_null(""),
            # department, category -> __UNKNOWN__ if null or empty
            pl.when(pl.col('department').is_null() | (pl.col('department') == ""))
              .then(pl.lit("__UNKNOWN__"))
              .otherwise(pl.col('department'))
              .alias('department'),
            pl.when(pl.col('category').is_null() | (pl.col('category') == ""))
              .then(pl.lit("__UNKNOWN__"))
              .otherwise(pl.col('category'))
              .alias('category')
        ])

        # 3. Cast numeric and datetime columns
        df_cleaned = df_cleaned.with_columns([
            pl.col('transaction_date').str.to_datetime(strict=False),
            pl.col('price').cast(pl.Float64, strict=False),
            pl.col('quantity').cast(pl.Float64, strict=False)
        ])

        # 4. Strict Validation Filters
        valid_df = df_cleaned.filter(
            pl.col('customer_id').is_not_null() & (pl.col('customer_id') != "") &
            pl.col('transaction_id').is_not_null() & (pl.col('transaction_id') != "") &
            pl.col('product_id').is_not_null() & (pl.col('product_id') != "") &
            pl.col('transaction_date').is_not_null() &
            pl.col('price').is_not_null() & pl.col('price').is_finite() & (pl.col('price') >= 0) &
            pl.col('quantity').is_not_null() & pl.col('quantity').is_finite() & (pl.col('quantity') > 0)
        )
        
        return valid_df

    def engineer_features(self, lazy_df: pl.LazyFrame) -> pl.DataFrame:
        """
        Calculates advanced RFM metrics using Polars aggregations.
        """
        # Calculate line total
        df = lazy_df.with_columns(
            (pl.col('quantity') * pl.col('price')).alias('line_total'),
            pl.col('transaction_date').dt.date().alias('tx_date_only')
        )

        # Collect just the max date to establish the reference date
        max_date = df.select(pl.col('transaction_date').max()).collect().item()
        # Add 1 day padding using Python datetime, then convert to Polars literal
        ref_date = max_date + pd.Timedelta(days=1)
        self.reference_date = ref_date
        
        # Primary RFM Aggregation GroupBy
        rfm = df.group_by("customer_id").agg([
            # Recency: Days since last purchase
            (pl.lit(ref_date) - pl.col('transaction_date').max()).dt.total_days().alias("Recency"),
            
            # Frequency: Unique transaction IDs
            pl.col('transaction_id').n_unique().alias("Frequency"),
            
            # Monetary: Total spend
            pl.col('line_total').sum().alias("Total_Spend"),
            
            # Tenure: Days between first and last purchase
            (pl.col('transaction_date').max() - pl.col('transaction_date').min()).dt.total_days().alias("Tenure"),
            
            # Diversity: Unique categories purchased
            pl.col('category').n_unique().alias("Unique_Categories"),
            
            # Inter-purchase Time setup: Get list of unique purchase dates per customer
            pl.col('tx_date_only').unique().sort().alias("purchase_dates")
        ])

        # Execute the lazy plan to get the RFM table in memory
        rfm_df = rfm.collect()
        
        # Calculate Average Inter-purchase Time using list expressions
        # We calculate the differences between consecutive dates in the list
        rfm_df = rfm_df.with_columns(
            pl.when(pl.col('purchase_dates').list.len() > 1)
            .then(
                pl.col('purchase_dates').list.eval(
                    pl.element().diff().dt.total_days()
                ).list.drop_nulls().list.mean()
            )
            .otherwise(pl.lit(0.0))
            .alias("Avg_InterPurchase_Days")
        )

        # Final cleanup: calculate AOV and drop the intermediate dates list
        rfm_final = rfm_df.with_columns(
            (pl.col("Total_Spend") / pl.col("Frequency")).alias("Avg_Order_Value")
        ).drop("purchase_dates")

        return rfm_final

    def _preprocess_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Applies Log1p transformation and Standardization."""
        numeric_cols = features_df.drop('customer_id', axis=1).columns
        data = features_df[numeric_cols].copy()
        
        for col in data.columns:
            data[col] = np.log1p(np.maximum(data[col], 0))
            
        scaled_array = self.scaler.fit_transform(data)
        return pd.DataFrame(scaled_array, columns=numeric_cols)

    def _find_optimal_k(self, scaled_data: pd.DataFrame) -> int:
        """Determines best k using Silhouette Score."""
        best_k = 2
        best_score = -1
        
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                
        return best_k

    def fit_predict(self, data_source) -> pd.DataFrame:
        """
        Executes the full pipeline. 
        Accepts either a file path (string) or an existing Polars LazyFrame/DataFrame.
        """
        # Ensure we are working with a LazyFrame for query optimization
        if isinstance(data_source, str):
            print(f"0. Scanning lazy data from {data_source}...")
            lazy_df = pl.scan_parquet(data_source)
        elif isinstance(data_source, pl.DataFrame):
            lazy_df = data_source.lazy()
        elif isinstance(data_source, pl.LazyFrame):
            lazy_df = data_source
        else:
            raise TypeError(
                f"Expected str, pl.DataFrame, or pl.LazyFrame; got {type(data_source).__name__}"
            )

        print("1. Validating and Cleaning Data (Lazy)...")
        clean_lazy = self.validate_and_clean(lazy_df)
        
        print("2. Executing Plan & Engineering Features...")
        polars_features = self.engineer_features(clean_lazy)
        print(f"   -> Generated features for {len(polars_features)} unique customers.")
        
        print("3. Preprocessing and Scaling Data...")
        # Convert Polars DataFrame to Pandas specifically for Scikit-Learn compatibility
        pandas_features = polars_features.to_pandas()
        scaled_data = self._preprocess_data(pandas_features)
        
        print("4. Optimizing Cluster Count...")
        self.optimal_k = self._find_optimal_k(scaled_data)
        print(f"   -> Optimal clusters found: {self.optimal_k}")
        
        print("5. Fitting Final Model...")
        self.model = KMeans(n_clusters=self.optimal_k, 
                            random_state=self.random_state, 
                            n_init=10)
        
        # Assign clusters back to the original pandas dataframe
        pandas_features['Cluster'] = self.model.fit_predict(scaled_data)
        return pandas_features


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Use actual clean transaction data
    data_path = "customer_transactions_clean.parquet"
    
    # Initialize and run the pipeline
    pipeline = PolarsCustomerSegmentation(max_clusters=8)
    
    # We pass the file path - it will be scanned as LazyFrame
    segmented_customers_pd = pipeline.fit_predict(data_path)
    
    # Profile the resulting clusters
    cluster_profiles = segmented_customers_pd.groupby('Cluster').mean(numeric_only=True).round(2)
    print("\nCluster Profiles (Averages):")
    print(cluster_profiles[['Recency', 'Frequency', 'Total_Spend', 'Avg_Order_Value']])
    
    # Save results
    segmented_customers_pd.to_parquet("customer_segments.parquet")
    print("\nSaved segments to customer_segments.parquet")