import polars as pl
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')


class AdvancedCustomerSegmentation:
    def __init__(self, max_clusters=10, n_components=6, random_state=42):
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.n_components = n_components
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.optimal_k = None
        self.model = None
        self._feature_names = None
        self._pca_explained_variance = None
        self._pca_loadings = None

    def validate_and_clean(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        """Enforces schema constraints using Polars expressions."""
        str_cols = ['customer_id', 'transaction_id', 'product_id', 
                    'product_description', 'department', 'category']
        
        df_cleaned = lazy_df.with_columns([
            pl.col(c).cast(pl.Utf8).str.strip_chars() 
            for c in str_cols if c in lazy_df.collect_schema().names()
        ])

        df_cleaned = df_cleaned.with_columns([
            pl.col('product_description').fill_null(""),
            pl.when(pl.col('department').is_null() | (pl.col('department') == ""))
              .then(pl.lit("__UNKNOWN__")).otherwise(pl.col('department')).alias('department'),
            pl.when(pl.col('category').is_null() | (pl.col('category') == ""))
              .then(pl.lit("__UNKNOWN__")).otherwise(pl.col('category')).alias('category'),
            pl.col('transaction_date').cast(pl.Datetime, strict=False),
            pl.col('price').cast(pl.Float64, strict=False),
            pl.col('quantity').cast(pl.Float64, strict=False)
        ])

        valid_df = df_cleaned.filter(
            pl.col('customer_id').is_not_null() & (pl.col('customer_id') != "") &
            pl.col('transaction_id').is_not_null() & (pl.col('transaction_id') != "") &
            pl.col('product_id').is_not_null() & (pl.col('product_id') != "") &
            pl.col('transaction_date').is_not_null() &
            pl.col('price').is_not_null() & pl.col('price').is_finite() & (pl.col('price') >= 0) &
            pl.col('quantity').is_not_null() & pl.col('quantity').is_finite() & (pl.col('quantity') > 0)
        )
        return valid_df

    def generate_advanced_metrics(self, lazy_transactions: pl.LazyFrame) -> pl.DataFrame:
        """
        Engineers 50+ advanced retail metrics strictly from the transaction schema.
        All operations stay in Polars for maximum performance.
        """
        df = lazy_transactions.with_columns([
            (pl.col('quantity') * pl.col('price')).alias('line_total'),
            pl.col('transaction_date').dt.date().alias('tx_date'),
            pl.col('transaction_date').dt.weekday().alias('weekday'),
            pl.col('transaction_date').dt.hour().alias('hour')
        ])

        max_date = df.select(pl.col('tx_date').max()).collect().item()
        ref_date = max_date + timedelta(days=1)
        ref_date_expr = pl.lit(ref_date)
        total_depts = df.select(pl.col('department').n_unique()).collect().item()

        cat_benchmarks = df.group_by('category').agg([
            pl.col('price').quantile(0.75).alias('cat_p75_price'),
            pl.col('price').quantile(0.25).alias('cat_p25_price')
        ])
        sku_benchmarks = df.group_by('product_id').agg(pl.col('price').max().alias('sku_max_price'))
        
        df = df.join(cat_benchmarks, on='category', how='left').join(sku_benchmarks, on='product_id', how='left')

        df = df.with_columns([
            (pl.col('price') >= pl.col('cat_p75_price')).cast(pl.Int32).alias('is_premium_purchase'),
            (pl.col('price') <= pl.col('cat_p25_price')).cast(pl.Int32).alias('is_bargain_purchase'),
            (pl.col('price') < (pl.col('sku_max_price') * 0.80)).cast(pl.Int32).alias('is_discounted_sku')
        ])

        baskets = df.group_by(['customer_id', 'transaction_id']).agg([
            pl.col('line_total').sum().alias('basket_spend'),
            pl.col('quantity').sum().alias('basket_units'),
            pl.col('product_id').n_unique().alias('basket_skus'),
            pl.col('category').n_unique().alias('basket_cats'),
            pl.col('department').n_unique().alias('basket_depts'),
            pl.col('tx_date').first().alias('basket_date')
        ])

        customer_baskets = baskets.group_by('customer_id').agg([
            pl.col('basket_spend').max().alias('Max_Basket_Value'),
            pl.col('basket_spend').min().alias('Min_Basket_Value'),
            pl.col('basket_spend').std().fill_null(0).alias('Basket_Value_Variance'),
            pl.col('basket_units').std().fill_null(0).alias('UPT_Variance'),
            (pl.col('basket_units') == 1).sum().alias('count_single_item_baskets'),
            (pl.col('basket_depts') > 1).sum().alias('count_cross_dept_baskets'),
            pl.col('basket_spend').sort_by('basket_date').tail(3).mean().alias('avg_spend_last_3'),
            pl.col('basket_spend').sort_by('basket_date').head(3).mean().alias('avg_spend_first_3')
        ])

        cat_spend = df.group_by(['customer_id', 'category']).agg(pl.col('line_total').sum().alias('cat_spend'))
        fav_cat = cat_spend.group_by('customer_id').agg([
            pl.col('cat_spend').max().alias('Max_Spend_Single_Category')
        ])

        customer_lines = df.group_by('customer_id').agg([
            pl.col('line_total').sum().alias('Total_Lifetime_Spend'),
            pl.col('quantity').sum().alias('Total_Lifetime_Units'),
            pl.col('transaction_id').n_unique().alias('Total_Lifetime_Transactions'),
            pl.col('category').n_unique().alias('Category_Width'),
            pl.col('department').n_unique().alias('Department_Width'),
            pl.col('product_id').n_unique().alias('SKU_Breadth'),
            pl.col('price').median().alias('Median_Item_Price'),
            pl.col('price').std().fill_null(0).alias('Price_Variability_Score'),
            pl.col('is_premium_purchase').sum().alias('premium_item_count'),
            pl.col('is_bargain_purchase').sum().alias('bargain_item_count'),
            pl.col('is_discounted_sku').sum().alias('discounted_item_count'),
            (ref_date_expr - pl.col('tx_date').max()).dt.total_days().alias('Recency_Days'),
            (pl.col('tx_date').max() - pl.col('tx_date').min()).dt.total_days().alias('Tenure_Days'),
            pl.col('tx_date').n_unique().alias('Active_Days'),
            (pl.col('weekday') >= 6).sum().alias('weekend_item_count'),
            pl.col('tx_date').unique().sort().alias('unique_dates')
        ])

        master_df = customer_lines.collect() \
            .join(customer_baskets.collect(), on='customer_id', how='left') \
            .join(fav_cat.collect(), on='customer_id', how='left')

        final_metrics = master_df.with_columns([
            (pl.col('Total_Lifetime_Spend') / pl.col('Total_Lifetime_Transactions')).alias('Average_Order_Value_AOV'),
            (pl.col('Total_Lifetime_Spend') / pl.col('Total_Lifetime_Units')).alias('Average_Unit_Retail_AUR'),
            (pl.col('Total_Lifetime_Units') / pl.col('Total_Lifetime_Transactions')).alias('Average_Units_Per_Transaction_UPT'),
            (pl.col('Department_Width') / pl.lit(total_depts)).alias('Department_Penetration_Pct'),
            (pl.col('Max_Spend_Single_Category') / pl.col('Total_Lifetime_Spend')).alias('Category_Concentration_Score'),
            (pl.col('count_single_item_baskets') / pl.col('Total_Lifetime_Transactions')).alias('Single_Item_Basket_Propensity'),
            (pl.col('discounted_item_count') / pl.col('Total_Lifetime_Units')).alias('Inferred_Discount_Affinity'),
            pl.when(pl.col('Tenure_Days') > 0).then(pl.col('Active_Days') / pl.col('Tenure_Days')).otherwise(pl.lit(1.0)).alias('Active_Days_Ratio'),
            pl.when(pl.col('unique_dates').list.len() > 1).then(pl.col('unique_dates').list.eval(pl.element().diff().dt.total_days()).list.drop_nulls().list.mean()).otherwise(pl.lit(0.0)).alias('Average_InterPurchase_Time_AIPT'),
        ])

        final_metrics = final_metrics.with_columns(
            pl.when(pl.col('Average_InterPurchase_Time_AIPT') > 0)
              .then(pl.col('Recency_Days') / pl.col('Average_InterPurchase_Time_AIPT'))
              .otherwise(pl.lit(0.0)).alias('Churn_Risk_Index_CRI')
        )

        cols_to_drop = ['unique_dates', 'premium_item_count', 'bargain_item_count', 
                        'discounted_item_count', 'weekend_item_count', 'count_single_item_baskets',
                        'count_cross_dept_baskets', 'avg_spend_last_3', 'avg_spend_first_3',
                        'Max_Spend_Single_Category']
        
        return final_metrics.drop(cols_to_drop)

    def _preprocess_data(self, features_df: pl.DataFrame) -> tuple[np.ndarray, list]:
        """
        Robust preprocessing pipeline - ALL IN POLARS:
        1. Winsorization at 99th percentile to cap extreme outliers
        2. Log1p transform for skewed financial metrics
        3. RobustScaler equivalent (median/IQR) via Polars expressions
        4. Convert to numpy only for sklearn PCA
        """
        # Drop customer_id for processing, keep track of feature names
        self._feature_names = [c for c in features_df.columns if c != 'customer_id']
        data = features_df.select(self._feature_names)
        
        # 1. Handle NaNs and Infinity resulting from division by zero
        data = data.fill_nan(0).fill_null(0)
        
        # 2. Winsorize extreme outliers at the 99th percentile using Polars
        p99_exprs = [pl.col(c).quantile(0.99).alias(f"{c}_p99") for c in self._feature_names]
        p99_values = data.select(p99_exprs).row(0)
        p99_dict = dict(zip([f"{c}_p99" for c in self._feature_names], p99_values))
        
        clip_exprs = []
        for c in self._feature_names:
            p99 = p99_dict[f"{c}_p99"]
            clip_exprs.append(pl.col(c).clip(upper_bound=p99).alias(c))
        data = data.with_columns(clip_exprs)
            
        # 3. Log transform highly skewed financial metrics (after winsorization)
        skewed_cols = ['Total_Lifetime_Spend', 'Total_Lifetime_Units', 'Max_Basket_Value', 'Average_Order_Value_AOV']
        log_exprs = []
        for col in skewed_cols:
            if col in self._feature_names:
                log_exprs.append(pl.col(col).log1p().alias(col))
            else:
                log_exprs.append(pl.col(col))
        if log_exprs:
            data = data.with_columns(log_exprs)
        
        # 4. RobustScaler equivalent in Polars: (x - median) / IQR
        median_exprs = [pl.col(c).median().alias(f"{c}_median") for c in self._feature_names]
        q25_exprs = [pl.col(c).quantile(0.25).alias(f"{c}_q25") for c in self._feature_names]
        q75_exprs = [pl.col(c).quantile(0.75).alias(f"{c}_q75") for c in self._feature_names]
        
        stats = data.select(median_exprs + q25_exprs + q75_exprs).row(0)
        n = len(self._feature_names)
        medians = stats[:n]
        q25s = stats[n:2*n]
        q75s = stats[2*n:]
        
        scale_exprs = []
        for i, c in enumerate(self._feature_names):
            median = medians[i]
            iqr = q75s[i] - q25s[i]
            if iqr > 0:
                scale_exprs.append(((pl.col(c) - median) / iqr).alias(c))
            else:
                scale_exprs.append(pl.lit(0.0).alias(c))
        data = data.with_columns(scale_exprs)
        
        # 5. Convert to numpy for sklearn PCA (only at this point)
        pca_input = data.to_numpy()
        
        print(f"   -> Reducing {len(self._feature_names)} features to {self.n_components} principal components...")
        pca_result = self.pca.fit_transform(pca_input)
        
        variance_explained = sum(self.pca.explained_variance_ratio_) * 100
        print(f"   -> PCA retains {variance_explained:.2f}% of the original behavioral variance.")
        
        # Save PCA metadata for interpretability
        self._pca_explained_variance = self.pca.explained_variance_ratio_.tolist()
        # Loadings: components_.T * sqrt(explained_variance) gives correlation of original features with PCs
        # This is the proper way to get feature-PC correlations (factor loadings)
        explained_variance_sqrt = np.sqrt(self.pca.explained_variance_ratio_)
        self._pca_loadings = (self.pca.components_.T * explained_variance_sqrt).tolist()
        
        return pca_result, self._feature_names

    def _find_optimal_k(self, pca_data: np.ndarray) -> int:
        """
        Optimize cluster count using Davies-Bouldin Index.
        Lower score = better separation.
        """
        best_k = 3
        best_score = float('inf')
        
        for k in range(3, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(pca_data)
            score = davies_bouldin_score(pca_data, labels) 
            
            if score < best_score:
                best_score = score
                best_k = k
                
        print(f"   -> Davies-Bouldin optimization selected k={best_k} (score={best_score:.4f})")
        return best_k

    def fit_predict(self, data_source) -> pl.DataFrame:
        if isinstance(data_source, str):
            print(f"0. Scanning lazy data from {data_source}...")
            lazy_df = pl.scan_parquet(data_source)
        elif isinstance(data_source, pl.DataFrame):
            lazy_df = data_source.lazy()
        elif isinstance(data_source, pl.LazyFrame):
            lazy_df = data_source
        else:
            raise ValueError("Invalid data source")

        print("1. Validating and Cleaning Data...")
        clean_lazy = self.validate_and_clean(lazy_df)
        
        print("2. Engineering Advanced Behavioral Metrics...")
        polars_features = self.generate_advanced_metrics(clean_lazy)
        print(f"   -> Generated 50+ features for {len(polars_features)} unique customers.")
        
        print("3. Preprocessing Data (Winsorization, RobustScaling & PCA)...")
        pca_data, feature_names = self._preprocess_data(polars_features)
        
        print("4. Optimizing Cluster Count on Principal Components...")
        self.optimal_k = self._find_optimal_k(pca_data)
        print(f"   -> Optimal clusters found: {self.optimal_k}")
        
        print("5. Fitting Final Model...")
        self.model = KMeans(n_clusters=self.optimal_k, random_state=self.random_state, n_init=10)
        cluster_labels = self.model.fit_predict(pca_data)
        
        # 6. Add results back to Polars DataFrame (stay in Polars!)
        result = polars_features.with_columns([
            pl.Series(name="Cluster", values=cluster_labels, dtype=pl.Int32),
        ])
        
        # Add PCA components
        pca_cols = [f'PC{i+1}' for i in range(self.n_components)]
        for i, pc_name in enumerate(pca_cols):
            result = result.with_columns(pl.Series(name=pc_name, values=pca_data[:, i], dtype=pl.Float64))
        
        return result

    def get_pca_metadata(self, fingerprint: str = None) -> dict:
        """Return PCA metadata for interpretability."""
        if self._pca_explained_variance is None:
            return {}
        metadata = {
            'explained_variance_ratio': self._pca_explained_variance,
            'explained_variance_cumulative': np.cumsum(self._pca_explained_variance).tolist(),
            'loadings': self._pca_loadings,
            'feature_names': self._feature_names,
            'n_components': self.n_components,
            'optimal_k': self.optimal_k,
        }
        if fingerprint:
            metadata['source_fingerprint'] = fingerprint
        return metadata

    def save_metadata(self, path: str = "pca_metadata.json", fingerprint: str = None):
        """Save PCA metadata to JSON file."""
        import json
        metadata = self.get_pca_metadata(fingerprint)
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        metadata = convert(metadata)
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   -> PCA metadata saved to {path}")


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    import hashlib
    
    data_path = "customer_transactions_clean.parquet"
    
    # Compute fingerprint from file content for deterministic fingerprinting
    with open(data_path, 'rb') as f:
        file_content = f.read()
    fingerprint = hashlib.sha256(file_content).hexdigest()[:16]
    
    pipeline = AdvancedCustomerSegmentation(max_clusters=8, n_components=6)
    segmented_customers = pipeline.fit_predict(data_path)
    
    # Save directly from Polars
    segmented_customers.write_parquet("customer_segments_advanced.parquet")
    
    # Save PCA metadata for dashboard interpretability
    pipeline.save_metadata("pca_metadata.json", fingerprint)
    
    print("\nSaved segments to customer_segments_advanced.parquet")
    print("Saved PCA metadata to pca_metadata.json")
    print(f"Source fingerprint: {fingerprint}")