import unittest

import pandas as pd

from purchase_inactivity import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features_at_snapshot,
    build_training_dataset,
    explain_customer,
    make_model,
    prepare_transactions,
)


def _tx(customer, date, transaction, price=10.0, quantity=1.0, category="A"):
    return {
        "customer_id": customer,
        "transaction_id": transaction,
        "transaction_date": date,
        "price": price,
        "quantity": quantity,
        "category": category,
    }


class PurchaseInactivityTests(unittest.TestCase):
    def setUp(self):
        rows = [
            # Both customers are active immediately before the first snapshot.
            _tx("A", "2024-01-01", "a1", 10),
            _tx("A", "2024-01-30", "a2", 20),
            # B purchases on the first target window; C buys on its first day.
            _tx("B", "2024-01-01", "b1", 10),
            _tx("B", "2024-01-30", "b2", 10),
            _tx("B", "2024-02-15", "b3", 30),
            _tx("C", "2024-01-01", "c1", 10),
            _tx("C", "2024-01-30", "c2", 10),
            _tx("C", "2024-01-31", "c3", 50),
            # Later purchases give the dataset more complete snapshots.
            _tx("A", "2024-04-15", "a3", 15),
            _tx("B", "2024-03-10", "b4", 12),
            _tx("C", "2024-04-02", "c4", 12),
        ]
        self.raw = pd.DataFrame(rows)
        self.prepared = prepare_transactions(self.raw)

    def test_snapshot_features_exclude_snapshot_day_and_future(self):
        features = build_features_at_snapshot(self.prepared, "2024-01-31")
        a = features.set_index("customer_id").loc["A"]
        # Jan 1 and Jan 30 are in the preceding 30-day window; Jan 31 is excluded.
        self.assertEqual(a["orders_30d"], 2)
        self.assertEqual(a["spend_30d"], 30.0)
        self.assertEqual(a["days_since_last_purchase"], 1)

    def test_training_labels_use_next_30_days(self):
        dataset = build_training_dataset(self.raw, "2024-05-31")
        first_snapshot = dataset.loc[dataset["snapshot_date"].eq(pd.Timestamp("2024-01-31"))].set_index("customer_id")
        self.assertEqual(int(first_snapshot.loc["A", TARGET_COLUMN]), 1)  # no buy Jan 31-Feb 29
        self.assertEqual(int(first_snapshot.loc["B", TARGET_COLUMN]), 0)  # bought Feb 15
        self.assertEqual(int(first_snapshot.loc["C", TARGET_COLUMN]), 0)  # bought on snapshot day

    def test_only_customers_active_in_previous_30_days_are_labeled(self):
        rows = [
            _tx("active", "2024-01-29", "o1"),
            _tx("dormant", "2023-11-01", "o2"),
            _tx("active", "2024-03-15", "o3"),
        ]
        ds = build_training_dataset(pd.DataFrame(rows), "2024-04-30")
        first = ds.loc[ds["snapshot_date"].eq(pd.Timestamp("2024-01-30"))]
        self.assertIn("active", first["customer_id"].tolist())
        self.assertNotIn("dormant", first["customer_id"].tolist())

    def test_local_explanations_sum_to_logistic_decision_score(self):
        import numpy as np

        rng = np.random.default_rng(7)
        x = pd.DataFrame(rng.normal(size=(80, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        y = (x["days_since_last_purchase"] + 0.6 * x["spend_change_30d"] < 0).astype(int)
        model = make_model().fit(x, y)
        current = x.iloc[[0, 1]].copy()
        current.insert(0, "customer_id", ["x", "y"])
        explanation = explain_customer(model, current, "x", top_n=len(FEATURE_COLUMNS))
        transformed = model[:-1].transform(current.loc[current["customer_id"].eq("x"), FEATURE_COLUMNS])
        decision = model.named_steps["classifier"].decision_function(transformed)[0]
        contribution_sum = explanation["contribution_log_odds"].sum()
        self.assertAlmostEqual(
            contribution_sum + model.named_steps["classifier"].intercept_[0],
            decision,
            places=6,
        )

    def test_rejects_overlapping_target_windows(self):
        with self.assertRaises(ValueError):
            build_training_dataset(self.raw, "2024-05-31", snapshot_step_days=14)

    def test_rejects_incomplete_date_before_observed_transactions(self):
        with self.assertRaises(ValueError):
            build_training_dataset(self.raw, "2024-01-15")


if __name__ == "__main__":
    unittest.main()
