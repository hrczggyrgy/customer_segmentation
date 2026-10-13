import unittest
import numpy as np
import pandas as pd
from frontend_storytelling import prepare_customers, segment_profiles, profile_matrix, selected_customer_ids


class StorytellingTests(unittest.TestCase):
    def setUp(self):
        self.customers = prepare_customers(pd.DataFrame({
            'customer_id': [1, 2, 3, 4], 'Cluster': [0, 0, 1, 1],
            'Total_Lifetime_Spend': [100., 100., 300., 500.],
            'Average_Order_Value_AOV': [10., 10., 10., 10.],
            'Recency_Days': [10., 10., 30., 30.],
            'Average_InterPurchase_Time_AIPT': [10., 10., 10., 0.],
            'Churn_Risk_Index_CRI': [1., 1., 3., 0.],
        }))

    def test_full_base_shares_and_order(self):
        profiles = segment_profiles(self.customers)
        self.assertEqual(profiles.index.tolist(), [1, 0])
        self.assertAlmostEqual(profiles.loc[1, 'spend_share'], 80.)
        self.assertAlmostEqual(profiles.loc[[1], 'customer_share'].sum(), 50.)
        self.assertAlmostEqual(profiles['customer_share'].sum(), 100.)

    def test_repeat_history_exclusion(self):
        profiles = segment_profiles(self.customers)
        self.assertEqual(profiles.loc[1, 'median_delay'], 3.)
        self.assertEqual(profiles.loc[1, 'repeat_share'], .5)
        self.assertIn('overdue repeat buyers', profiles.loc[1, 'label'])
        self.customers['Average_InterPurchase_Time_AIPT'] = 0
        self.assertTrue(segment_profiles(self.customers)['label'].str.contains('limited repeat history').all())

    def test_heatmap_alignment_and_constant_metric(self):
        metrics, z, custom = profile_matrix(self.customers, [1, 0])
        self.assertEqual(z.index.tolist(), [1, 0])
        self.assertEqual(z['Average_Order_Value_AOV'].tolist(), [0., 0.])
        i = metrics.index('Total_Lifetime_Spend')
        self.assertEqual(custom[0, i, 0], 400.)
        self.assertEqual(custom[0, i, 1], 250.)
        self.assertTrue(np.isfinite(z.to_numpy()).all())

    def test_selection_uses_ids_not_indices(self):
        points = [{'point_index': 0, 'customdata': ['3']}, {'point_index': 0, 'customdata': ['1']}, {}]
        self.assertEqual(selected_customer_ids(points), {'1', '3'})

    def test_duplicate_ids_rejected(self):
        frame = self.customers.copy()
        frame.loc[1, 'customer_id'] = '1'
        with self.assertRaises(ValueError):
            prepare_customers(frame)

    def test_zero_spend_and_missing_columns(self):
        self.customers['Total_Lifetime_Spend'] = 0
        self.assertEqual(segment_profiles(self.customers)['spend_share'].sum(), 0)
        with self.assertRaises(ValueError):
            prepare_customers(self.customers.drop(columns='Cluster'))


if __name__ == '__main__':
    unittest.main()
