import ast
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from datetime import date, datetime
from data_safety import source_fingerprint, generate_session_segments, finalized_month_end, period_is_finalized, retention_matrix


class SafetyTests(unittest.TestCase):
    def test_identical_upload_bytes_and_local_hash(self):
        content = b'abc'
        stream = io.BytesIO(content)
        stream.seek(2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.parquet'
            path.write_bytes(content)
            self.assertEqual(source_fingerprint(stream), source_fingerprint(path))
        self.assertEqual(stream.tell(), 2)
        self.assertEqual(source_fingerprint(content), hashlib.sha256(content).hexdigest())
        self.assertEqual(len(source_fingerprint(content)), 64)

    def test_chunking_and_changed_data(self):
        content = b'a' * (1024 * 1024 + 17)
        self.assertEqual(source_fingerprint(io.BytesIO(content)), source_fingerprint(content))
        self.assertNotEqual(source_fingerprint(content), source_fingerprint(content + b'b'))

    def test_sessions_reuse_and_force_are_isolated(self):
        calls = []
        class Pipeline:
            def fit_predict(self, data):
                calls.append(data)
                return {'source': data, 'run': len(calls)}
            def get_pca_metadata(self, fingerprint):
                return {'source_fingerprint': fingerprint}
        a, b = {}, {}
        fa, fb = source_fingerprint(b'A'), source_fingerprint(b'B')
        first, _ = generate_session_segments(a, 'A', fa, Pipeline)
        generate_session_segments(b, 'B', fb, Pipeline)
        again, _ = generate_session_segments(a, 'A', fa, Pipeline)
        self.assertIs(first, again)
        self.assertEqual(calls, ['A', 'B'])
        generate_session_segments(a, 'A', fa, Pipeline, force=True)
        self.assertEqual(b['segments_df']['source'], 'B')
        generate_session_segments(a, 'B', fb, Pipeline)
        self.assertEqual(a['segments_fingerprint'], fb)
        self.assertEqual(calls, ['A', 'B', 'A', 'B'])

    def test_failure_does_not_publish_partial_pair(self):
        class Pipeline:
            def fit_predict(self, data):
                return 'new segments'
            def get_pca_metadata(self, fingerprint):
                raise RuntimeError('metadata failed')
        state = {'segments_df': 'old', 'pca_metadata': {'source_fingerprint': 'old'}}
        before = state.copy()
        with self.assertRaises(RuntimeError):
            generate_session_segments(state, 'data', source_fingerprint(b'data'), Pipeline)
        self.assertEqual(state, before)
        with self.assertRaises(ValueError):
            generate_session_segments({}, 'data', '1234567890abcdef', Pipeline)

    def test_partial_end_of_month_and_year_boundaries(self):
        self.assertEqual(finalized_month_end(date(2024, 3, 15)), date(2024, 2, 29))
        self.assertEqual(finalized_month_end(date(2024, 3, 31)), date(2024, 2, 29))
        self.assertEqual(finalized_month_end(datetime(2025, 1, 1)), date(2024, 12, 31))

    def test_explicit_completeness(self):
        latest = date(2024, 3, 15)
        self.assertEqual(finalized_month_end(latest, date(2024, 3, 31)), date(2024, 3, 31))
        self.assertEqual(finalized_month_end(latest, date(2024, 3, 14)), date(2024, 2, 29))
        with self.assertRaises(ValueError):
            finalized_month_end(latest, date(2024, 4, 30))

    def test_period_eligibility(self):
        self.assertTrue(period_is_finalized(date(2023, 12, 1), 1, date(2024, 1, 31)))
        self.assertFalse(period_is_finalized(date(2023, 12, 1), 2, date(2024, 1, 31)))
        self.assertFalse(period_is_finalized(date(2024, 1, 1), -1, date(2024, 1, 31)))

    def test_app_has_no_shared_result_cache_or_artifact_io(self):
        source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
        ast.parse(source)
        for forbidden in ['@st.cache_data', 'write_parquet', 'pca_metadata.json', 'customer_segments_advanced.parquet', '.clear()']:
            self.assertNotIn(forbidden, source)
        self.assertIn('source_fingerprint(uploaded_transactions)', source)
        # Check that demo file fingerprints are used
        self.assertIn('source_fingerprint(demo_path)', source)
        standalone = (Path(__file__).resolve().parents[1] / 'segmentation.py').read_text()
        self.assertIn('fingerprint = source_fingerprint(data_path)', standalone)


@unittest.skipUnless(importlib.util.find_spec('polars'), 'Polars required for retention integration tests')
class RetentionTests(unittest.TestCase):
    def matrix(self, rows, cutoff):
        import polars as pl
        frame = pl.DataFrame({'customer_id': [r[0] for r in rows], 'transaction_date': [datetime.fromisoformat(r[1]) for r in rows]})
        return retention_matrix(frame.lazy(), cutoff)

    def test_partial_month_excluded_and_future_null(self):
        result = self.matrix([('A', '2024-01-01'), ('B', '2024-01-10'), ('C', '2024-02-01'), ('A', '2024-03-15'), ('D', '2024-03-01')], date(2024, 2, 29))
        self.assertEqual(result['Cohort'].to_list(), ['2024-01', '2024-02'])
        self.assertEqual(result['1'].to_list(), [0., None])
        self.assertEqual(result['0'].to_list(), [1., 1.])

    def test_missing_period_becomes_observed_zero(self):
        result = self.matrix([('A', '2024-01-01'), ('B', '2024-03-15')], date(2024, 2, 29))
        self.assertEqual(result['1'].to_list(), [0.])

    def test_explicit_complete_month_included(self):
        result = self.matrix([('A', '2024-01-01'), ('B', '2024-01-10'), ('A', '2024-03-15')], date(2024, 3, 31))
        self.assertEqual(result['2'].to_list(), [.5])

    def test_no_eligible_cohort(self):
        result = self.matrix([('A', '2024-03-15')], date(2024, 2, 29))
        self.assertEqual(result.height, 0)


if __name__ == '__main__':
    unittest.main()
