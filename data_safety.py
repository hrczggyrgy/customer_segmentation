"""Dataset identity, session-only results, and conservative retention boundaries."""
import calendar
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path


def source_fingerprint(source):
    """Full SHA-256 of file bytes; seekable streams retain their position."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return hashlib.sha256(source).hexdigest()
    if isinstance(source, (str, Path)):
        with open(source, 'rb') as stream:
            return source_fingerprint(stream)
    position = source.tell()
    digest = hashlib.sha256()
    try:
        source.seek(0)
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    finally:
        source.seek(position)
    return digest.hexdigest()


def generate_session_segments(state, transactions, fingerprint, factory, force=False):
    """No shared cache or disk artifacts. Publish the result pair only on success."""
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
        raise ValueError('Expected a full lowercase SHA-256 fingerprint.')
    metadata = state.get('pca_metadata')
    if (not force and state.get('segments_generated') and state.get('segments_df') is not None
            and state.get('segments_fingerprint') == fingerprint
            and metadata and metadata.get('source_fingerprint') == fingerprint):
        return state['segments_df'], metadata
    pipeline = factory()
    segments = pipeline.fit_predict(transactions)
    metadata = pipeline.get_pca_metadata(fingerprint)
    if not metadata or metadata.get('source_fingerprint') != fingerprint:
        raise ValueError('Pipeline metadata does not match the source fingerprint.')
    state.update(segments_df=segments, pca_metadata=metadata,
                 segments_fingerprint=fingerprint, segments_generated=True)
    return segments, metadata


def as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError('Expected a date or datetime.')


def month_end(value):
    value = as_date(value)
    return value.replace(day=calendar.monthrange(value.year, value.month)[1])


def finalized_month_end(latest_observed, complete_through=None):
    """By default exclude the latest observed month, even with a month-end sale."""
    latest = as_date(latest_observed)
    if complete_through is None:
        return latest.replace(day=1) - timedelta(days=1)
    complete = as_date(complete_through)
    if complete > month_end(latest):
        raise ValueError('Completeness cannot extend beyond the latest observed month.')
    return complete if complete == month_end(complete) else complete.replace(day=1) - timedelta(days=1)


def period_is_finalized(cohort, period, cutoff):
    if period < 0:
        return False
    cohort, cutoff = as_date(cohort), as_date(cutoff)
    return cohort.year * 12 + cohort.month + period <= cutoff.year * 12 + cutoff.month


def retention_matrix(transactions_lf, cutoff):
    """Polars retention; unfinished periods are null, observed absences are zero."""
    import polars as pl
    cutoff = as_date(cutoff)
    if cutoff != month_end(cutoff):
        raise ValueError('Retention cutoff must be a month-end date.')
    cutoff_index = cutoff.year * 12 + cutoff.month
    df = transactions_lf.with_columns(pl.col('transaction_date').dt.truncate('1mo').alias('OrderMonth'))
    cohorts = df.group_by('customer_id').agg(pl.col('OrderMonth').min().alias('CohortMonth'))
    df = df.join(cohorts, on='customer_id', how='left').with_columns(
        ((pl.col('OrderMonth').dt.year() - pl.col('CohortMonth').dt.year()) * 12
         + pl.col('OrderMonth').dt.month() - pl.col('CohortMonth').dt.month()).alias('PeriodNumber')
    ).filter((pl.col('CohortMonth').dt.year() * 12 + pl.col('CohortMonth').dt.month()) <= cutoff_index)
    counts = df.group_by(['CohortMonth', 'PeriodNumber']).agg(
        pl.col('customer_id').n_unique().alias('customer_count')
    ).collect()
    if counts.is_empty():
        return pl.DataFrame(schema={'Cohort': pl.String})
    pivot = counts.pivot(values='customer_count', index='CohortMonth', on='PeriodNumber',
                         aggregate_function='first').sort('CohortMonth')
    earliest = pivot['CohortMonth'].min()
    max_period = cutoff_index - (earliest.year * 12 + earliest.month)
    periods = [str(p) for p in range(max_period + 1)]
    for column in periods:
        if column not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    pivot = pivot.with_columns(pl.col('0').alias('_cohort_size'))
    expressions = []
    for p, column in enumerate(periods):
        expressions.append(pl.when(
            (pl.col('_cohort_size') > 0) &
            ((pl.col('CohortMonth').dt.year() * 12 + pl.col('CohortMonth').dt.month() + p) <= cutoff_index)
        ).then(pl.col(column).fill_null(0) / pl.col('_cohort_size')).otherwise(None).alias(column))
    return pivot.with_columns(expressions).with_columns(
        pl.col('CohortMonth').dt.strftime('%Y-%m').alias('Cohort')
    ).select(['Cohort'] + periods)
