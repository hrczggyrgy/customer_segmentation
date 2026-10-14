import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from frontend_storytelling import contribution_metrics, segment_differentiators, render_contribution, render_profiles, segment_profiles


class InsightTests(unittest.TestCase):
    def setUp(self):
        x = np.arange(1., 9.)
        self.customers = pd.DataFrame({
            'customer_id': [str(i) for i in range(8)], 'Cluster': [0] * 4 + [1] * 4,
            'Total_Lifetime_Spend': x * 10, 'Recency_Days': 9 - x,
            'Average_Order_Value_AOV': 20., 'Average_InterPurchase_Time_AIPT': 2.,
            'Churn_Risk_Index_CRI': 1., 'Total_Lifetime_Transactions': x,
        })

    def test_contribution_example(self):
        self.assertEqual(contribution_metrics(20, 35, 1000), (1.75, 15.))

    def test_zero_spend_share_is_valid_with_positive_base(self):
        self.assertEqual(contribution_metrics(20, 0, 1000), (0., -20.))

    def test_undefined_contribution(self):
        for args in [(0, 35, 1000), (20, 35, 0), (20, 35, -1),
                     (np.nan, 35, 1000), (20, np.inf, 1000), (20, 35, np.inf),
                     (None, 35, 1000), (20, -1, 1000)]:
            with self.subTest(args=args):
                self.assertEqual(contribution_metrics(*args), (None, None))

    def test_negative_gap(self):
        self.assertEqual(contribution_metrics(50, 20, 1000), (.4, -30.))

    def test_ranking_and_signed_direction(self):
        traits = segment_differentiators(self.customers, [1])[1]
        self.assertEqual(len(traits), 3)
        magnitudes = [abs(t['z_score']) for t in traits]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
        recency = next(t for t in traits if t['metric'] == 'Recency_Days')
        self.assertEqual(recency['direction'], 'below')
        self.assertEqual(recency['segment_mean'], 2.5)
        self.assertEqual(recency['overall_mean'], 4.5)

    def test_trait_limit(self):
        self.assertEqual(len(segment_differentiators(self.customers, [1], max_traits=2)[1]), 2)

    def test_constant_and_missing_metrics_excluded(self):
        frame = self.customers.copy()
        frame['Recency_Days'] = np.nan
        traits = segment_differentiators(frame, [1], max_traits=8)[1]
        names = [t['metric'] for t in traits]
        self.assertNotIn('Average_Order_Value_AOV', names)
        self.assertNotIn('Recency_Days', names)
        self.assertNotIn('Category_Width', names)
        for metric in frame.columns.difference(['customer_id', 'Cluster']):
            frame[metric] = 1.
        self.assertEqual(segment_differentiators(frame, [1])[1], [])

    def test_threshold_suppresses_small_differences(self):
        frame = self.customers.copy()
        frame['Total_Lifetime_Spend'] = [0, 1000, 0, 1000, 1, 1001, 1, 1001]
        traits = segment_differentiators(frame, [1], max_traits=8)[1]
        self.assertNotIn('Total_Lifetime_Spend', [t['metric'] for t in traits])
        self.assertEqual(segment_differentiators(frame, [1], min_abs_z=10)[1], [])

    def test_filter_does_not_change_full_base_benchmarks(self):
        all_traits = segment_differentiators(self.customers, [0, 1])
        self.assertEqual(all_traits[1], segment_differentiators(self.customers, [1])[1])
        profiles = segment_profiles(self.customers)
        row = profiles.loc[1]
        self.assertEqual(row['customer_share'], 50.)
        self.assertAlmostEqual(contribution_metrics(row['customer_share'], row['spend_share'], profiles['spend'].sum())[0], 65 / 45)

    def test_recency_wording_is_feature_aware(self):
        for cluster, expected in [(0, 'Longer since'), (1, 'More recent')]:
            traits = segment_differentiators(self.customers, [cluster])[cluster]
            recency = next(t for t in traits if t['metric'] == 'Recency_Days')
            self.assertTrue(recency['context'].startswith(expected))

    def test_invalid_parameters(self):
        for kwargs in [{'max_traits': 0}, {'max_traits': 1.5}, {'min_abs_z': -1}, {'min_abs_z': np.nan}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                segment_differentiators(self.customers, [1], **kwargs)

    def test_helpers_do_not_mutate_input(self):
        before = self.customers.copy(deep=True)
        segment_differentiators(self.customers, [1])
        pd.testing.assert_frame_equal(self.customers, before)

    def test_ties_follow_metric_order(self):
        frame = self.customers.copy()
        frame['Total_Lifetime_Spend'] = frame['Total_Lifetime_Transactions']
        frame['Recency_Days'] = frame['Total_Lifetime_Transactions']
        traits = segment_differentiators(frame, [1], max_traits=2)[1]
        self.assertEqual([t['metric'] for t in traits], ['Total_Lifetime_Spend', 'Recency_Days'])


class RenderingTests(unittest.TestCase):
    def setUp(self):
        InsightTests.setUp(self)
        self.metrics, self.captions = [], []
        metrics, captions = self.metrics, self.captions
        class Column:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def metric(self, label, value, **kwargs):
                metrics.append((label, value))
        self.st = types.ModuleType('streamlit')
        self.st.columns = lambda n: [Column() for _ in range(n)]
        self.st.caption = lambda text: captions.append(text)
        for name in ['subheader', 'header', 'info', 'write', 'dataframe', 'markdown']:
            setattr(self.st, name, lambda *args, **kwargs: None)
        self.st.plotly_chart = lambda fig, **kwargs: fig.to_plotly_json()
        self.st.column_config = types.SimpleNamespace(NumberColumn=lambda *args, **kwargs: kwargs)
        self.profiles = segment_profiles(self.customers)
        self.colors = {label: color for label, color in zip(self.profiles['label'], ['#4477AA', '#EE6677'])}

    def test_contribution_cards_render_with_full_base_scope(self):
        with patch.dict(sys.modules, {'streamlit': self.st}):
            render_contribution(self.profiles, [1], self.colors)
        self.assertEqual(self.metrics, [('Spend contribution index', '1.44×'), ('Gap vs. customer share', '+22.2 pp')])
        self.assertTrue(any('among selected segments' in text for text in self.captions))

    def test_undefined_cards_render_na(self):
        profiles = self.profiles.copy()
        profiles['spend'] = profiles['spend_share'] = 0.
        with patch.dict(sys.modules, {'streamlit': self.st}):
            render_contribution(profiles, [1], self.colors)
        self.assertEqual([value for _, value in self.metrics], ['N/A', 'N/A'])

    def test_profile_narrative_renders_and_is_qualified(self):
        with patch.dict(sys.modules, {'streamlit': self.st}):
            render_profiles(self.customers, self.profiles, [1])
        self.assertTrue(any('More recent last observed purchase' in text for text in self.captions))
        self.assertTrue(any('not statistical significance' in text for text in self.captions))


if __name__ == '__main__':
    unittest.main()
