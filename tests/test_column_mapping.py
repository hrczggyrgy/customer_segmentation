import unittest
import polars as pl

# Copy the functions here to avoid importing app (which runs Streamlit code)
REQUIRED_TX_COLUMNS = {
    "customer_id", "transaction_id", "product_id",
    "product_description", "department", "category",
    "transaction_date", "price", "quantity",
}

COLUMN_ALIASES = {
    "customer_id": ["customer_id", "customerid", "cust_id", "client_id", "user_id", "buyer_id"],
    "transaction_id": ["transaction_id", "transactionid", "txn_id", "order_id", "orderid", "invoice_id"],
    "product_id": ["product_id", "productid", "sku", "item_id", "itemid"],
    "product_description": ["product_description", "productdescription", "description", "product_name", "productname", "item_description"],
    "department": ["department", "dept", "division", "category_group"],
    "category": ["category", "cat", "product_category", "subcategory"],
    "transaction_date": ["transaction_date", "transactiondate", "date", "order_date", "orderdate", "purchase_date", "purchasedate", "txn_date"],
    "price": ["price", "unit_price", "unitprice", "amount", "cost", "sale_price"],
    "quantity": ["quantity", "qty", "units", "quantity_sold", "units_sold"],
}


def suggest_column_mapping(df_columns: list[str]) -> dict[str, str]:
    mapping = {}
    df_cols_lower = {c.lower(): c for c in df_columns}
    for required_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in df_cols_lower:
                mapping[required_col] = df_cols_lower[alias.lower()]
                break
    return mapping


def apply_column_mapping(df: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    rename_dict = {v: k for k, v in mapping.items() if v in df.columns}
    if rename_dict:
        df = df.rename(rename_dict)
    return df


class ColumnMappingTests(unittest.TestCase):
    def test_suggest_column_mapping_exact_match(self):
        df_cols = ["customer_id", "transaction_id", "product_id", 
                   "product_description", "department", "category",
                   "transaction_date", "price", "quantity"]
        mapping = suggest_column_mapping(df_cols)
        self.assertEqual(mapping, {c: c for c in REQUIRED_TX_COLUMNS})

    def test_suggest_column_mapping_common_aliases(self):
        df_cols = ["cust_id", "order_id", "sku", "product_name", 
                   "dept", "cat", "order_date", "unit_price", "qty"]
        mapping = suggest_column_mapping(df_cols)
        expected = {
            "customer_id": "cust_id",
            "transaction_id": "order_id",
            "product_id": "sku",
            "product_description": "product_name",
            "department": "dept",
            "category": "cat",
            "transaction_date": "order_date",
            "price": "unit_price",
            "quantity": "qty",
        }
        self.assertEqual(mapping, expected)

    def test_suggest_column_mapping_case_insensitive(self):
        df_cols = ["CUSTOMER_ID", "Transaction_ID", "Product_ID"]
        mapping = suggest_column_mapping(df_cols)
        self.assertEqual(mapping["customer_id"], "CUSTOMER_ID")
        self.assertEqual(mapping["transaction_id"], "Transaction_ID")
        self.assertEqual(mapping["product_id"], "Product_ID")

    def test_suggest_column_mapping_partial(self):
        df_cols = ["customer_id", "transaction_id", "product_id", "price", "quantity"]
        mapping = suggest_column_mapping(df_cols)
        self.assertEqual(set(mapping.keys()), {"customer_id", "transaction_id", "product_id", "price", "quantity"})

    def test_apply_column_mapping(self):
        df = pl.DataFrame({
            "cust_id": ["1", "2"],
            "order_id": ["a", "b"],
            "sku": ["x", "y"],
            "order_date": ["2024-01-01", "2024-01-02"],
            "unit_price": [10.0, 20.0],
            "qty": [1, 2],
        })
        mapping = {
            "customer_id": "cust_id",
            "transaction_id": "order_id",
            "product_id": "sku",
            "transaction_date": "order_date",
            "price": "unit_price",
            "quantity": "qty",
        }
        result = apply_column_mapping(df, mapping)
        self.assertEqual(set(result.columns), set(mapping.keys()))

    def test_apply_column_mapping_ignores_missing(self):
        df = pl.DataFrame({"customer_id": ["1"], "transaction_id": ["a"]})
        mapping = {"customer_id": "customer_id", "transaction_id": "transaction_id", "product_id": "sku"}
        result = apply_column_mapping(df, mapping)
        self.assertEqual(set(result.columns), {"customer_id", "transaction_id"})


if __name__ == "__main__":
    unittest.main()