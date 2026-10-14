"""Segment Migration Sankey: three historical snapshots with a frozen segmentation model.

This module builds a customer migration Sankey diagram by:
1. Automatically selecting three snapshot dates from a date range
2. Reconstructing customer features at each snapshot using equal-length windows
3. Fitting a segmentation model on snapshot 1 and freezing its parameters
4. Scoring snapshots 2 and 3 against the frozen model
5. Aggregating transitions and rendering an interactive Plotly Sankey diagram
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from segmentation import AdvancedCustomerSegmentation


@dataclass(frozen=True)
class MigrationConfig:
    """Configuration for segment migration analysis."""
    min_range_days: int = 3
    max_window_days: int = 90
    min_customers_snapshot1: int = 30
    random_state: int = 42
    max_clusters: int = 8
    n_components: int = 6


@dataclass
class FrozenPreprocessing:
    """Frozen preprocessing parameters from snapshot 1."""
    feature_names: list[str]
    p99_thresholds: dict[str, float]
    skewed_cols: list[str]
    medians: np.ndarray
    iqrs: np.ndarray
    pca: PCA
    kmeans: KMeans
    optimal_k: int
    
    def transform(self, features_df: pl.DataFrame) -> np.ndarray:
        """Apply frozen preprocessing to new features and return PCA-transformed data."""
        data = features_df.select(self.feature_names)
        
        # Fill NaNs and infinity
        data = data.fill_nan(0).fill_null(0)
        
        # Apply frozen winsorization thresholds
        clip_exprs = [
            pl.col(c).clip(upper_bound=self.p99_thresholds[c]).alias(c)
            for c in self.feature_names
        ]
        data = data.with_columns(clip_exprs)
        
        # Apply frozen log transform
        log_exprs = []
        for col in self.feature_names:
            if col in self.skewed_cols:
                log_exprs.append(pl.col(col).log1p().alias(col))
            else:
                log_exprs.append(pl.col(col))
        if log_exprs:
            data = data.with_columns(log_exprs)
        
        # Apply frozen RobustScaler (median/IQR)
        scale_exprs = []
        for i, c in enumerate(self.feature_names):
            median = self.medians[i]
            iqr = self.iqrs[i]
            if iqr > 0:
                scale_exprs.append(((pl.col(c) - median) / iqr).alias(c))
            else:
                scale_exprs.append(pl.lit(0.0).alias(c))
        data = data.with_columns(scale_exprs)
        
        # Convert to numpy and apply frozen PCA
        pca_input = data.to_numpy()
        return self.pca.transform(pca_input)


@dataclass
class SnapshotResult:
    """Results for a single snapshot."""
    cutoff_date: date
    window_start: date
    window_end: date
    features_df: pl.DataFrame
    pca_data: np.ndarray
    cluster_labels: np.ndarray
    n_customers: int


def select_snapshot_dates(
    tx_df: pl.DataFrame | pl.LazyFrame,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> tuple[date, date, date]:
    """Automatically select three snapshot dates from the transaction date range.
    
    Args:
        tx_df: Validated transaction DataFrame or LazyFrame with 'transaction_date' column
        start_date: Optional start of range (defaults to min transaction date)
        end_date: Optional end of range (defaults to max transaction date)
    
    Returns:
        Tuple of (snapshot1_date, snapshot2_date, snapshot3_date)
    
    Raises:
        ValueError: If date range is shorter than min_range_days
    """
    if isinstance(tx_df, pl.LazyFrame):
        min_dt = tx_df.select(pl.col('transaction_date').min()).collect().item()
        max_dt = tx_df.select(pl.col('transaction_date').max()).collect().item()
    else:
        min_dt = tx_df.select(pl.col('transaction_date').min()).item()
        max_dt = tx_df.select(pl.col('transaction_date').max()).item()
    
    if start_date is None:
        start_date = min_dt
    if end_date is None:
        end_date = max_dt
    
    range_days = (end_date - start_date).days
    if range_days < 3:
        raise ValueError(
            f"Date range must be at least 3 calendar days; got {range_days} days. "
            "Need at least 3 days to create three distinct snapshots."
        )
    
    # Place snapshots at end of each third
    third = range_days / 3
    snap1 = start_date + timedelta(days=int(third))
    snap2 = start_date + timedelta(days=int(2 * third))
    snap3 = end_date
    
    # Ensure distinct dates
    if snap1 >= snap2:
        snap1 = start_date + timedelta(days=1)
    if snap2 >= snap3:
        snap2 = start_date + timedelta(days=min(2, range_days - 1))
    
    return snap1, snap2, snap3


def compute_window_days(start_date: date, end_date: date, max_window_days: int = 90) -> int:
    """Compute equal lookback window length for all three snapshots.
    
    Uses min(90, floor(range_days / 3)).
    """
    range_days = (end_date - start_date).days
    return min(max_window_days, range_days // 3)


def build_features_at_cutoff(
    tx_df: pl.DataFrame,
    cutoff_date: date,
    window_days: int
) -> pl.DataFrame:
    """Generate advanced metrics for customers as of a specific cutoff date.
    
    Only uses transactions on or before the cutoff date. Lookback window
    is [cutoff - window_days, cutoff].
    
    Reuses the core feature engineering from AdvancedCustomerSegmentation
    but with a date filter applied.
    """
    window_start = cutoff_date - timedelta(days=window_days)
    window_end = cutoff_date
    
    # Filter transactions to the window
    lazy_tx = tx_df.filter(
        (pl.col('transaction_date') >= window_start) &
        (pl.col('transaction_date') <= window_end)
    ).lazy()
    
    # Reuse the segmentation's feature engineering
    seg = AdvancedCustomerSegmentation(
        max_clusters=8,
        n_components=6,
        random_state=42
    )
    
    # Validate and clean
    clean_lazy = seg.validate_and_clean(lazy_tx)
    
    # Generate metrics - this will use the window's max date as reference
    features = seg.generate_advanced_metrics(clean_lazy)
    
    # Add window metadata
    features = features.with_columns([
        pl.lit(window_start).alias('window_start'),
        pl.lit(window_end).alias('window_end'),
        pl.lit(cutoff_date).alias('cutoff_date'),
    ])
    
    return features


def fit_frozen_model(
    snapshot1_features: pl.DataFrame,
    config: MigrationConfig
) -> FrozenPreprocessing:
    """Fit the full preprocessing + PCA + KMeans pipeline on snapshot 1 and freeze parameters.
    
    This replicates AdvancedCustomerSegmentation._preprocess_data and _find_optimal_k
    but stores all fitted parameters for reuse.
    """
    # Get feature names (exclude customer_id and window metadata)
    exclude_cols = {'customer_id', 'window_start', 'window_end', 'cutoff_date'}
    feature_names = [c for c in snapshot1_features.columns if c not in exclude_cols]
    
    data = snapshot1_features.select(feature_names)
    
    # 1. Fill NaNs and infinity
    data = data.fill_nan(0).fill_null(0)
    
    # 2. Compute and store winsorization thresholds (99th percentile)
    p99_exprs = [pl.col(c).quantile(0.99).alias(f"{c}_p99") for c in feature_names]
    p99_values = data.select(p99_exprs).row(0)
    p99_dict = dict(zip([f"{c}_p99" for c in feature_names], p99_values))
    
    # Apply clipping
    clip_exprs = [
        pl.col(c).clip(upper_bound=p99_dict[f"{c}_p99"]).alias(c)
        for c in feature_names
    ]
    data = data.with_columns(clip_exprs)
    
    # 3. Log transform skewed financial metrics
    skewed_cols = ['Total_Lifetime_Spend', 'Total_Lifetime_Units', 'Max_Basket_Value', 'Average_Order_Value_AOV']
    skewed_cols = [c for c in skewed_cols if c in feature_names]
    
    log_exprs = []
    for col in feature_names:
        if col in skewed_cols:
            log_exprs.append(pl.col(col).log1p().alias(col))
        else:
            log_exprs.append(pl.col(col))
    if log_exprs:
        data = data.with_columns(log_exprs)
    
    # 4. Compute and store RobustScaler parameters (median, IQR)
    median_exprs = [pl.col(c).median().alias(f"{c}_median") for c in feature_names]
    q25_exprs = [pl.col(c).quantile(0.25).alias(f"{c}_q25") for c in feature_names]
    q75_exprs = [pl.col(c).quantile(0.75).alias(f"{c}_q75") for c in feature_names]
    
    stats = data.select(median_exprs + q25_exprs + q75_exprs).row(0)
    n = len(feature_names)
    medians = np.array(stats[:n], dtype=float)
    q25s = np.array(stats[n:2*n], dtype=float)
    q75s = np.array(stats[2*n:], dtype=float)
    iqrs = q75s - q25s
    
    # Apply scaling
    scale_exprs = []
    for i, c in enumerate(feature_names):
        median = medians[i]
        iqr = iqrs[i]
        if iqr > 0:
            scale_exprs.append(((pl.col(c) - median) / iqr).alias(c))
        else:
            scale_exprs.append(pl.lit(0.0).alias(c))
    data = data.with_columns(scale_exprs)
    
    # 5. Fit PCA on snapshot 1
    pca_input = data.to_numpy()
    pca = PCA(n_components=config.n_components, random_state=config.random_state)
    pca.fit(pca_input)
    pca_data = pca.transform(pca_input)
    
    # 6. Find optimal K and fit KMeans
    best_k = 3
    best_score = float('inf')
    from sklearn.metrics import davies_bouldin_score
    
    for k in range(3, config.max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=config.random_state, n_init=10)
        labels = kmeans.fit_predict(pca_data)
        score = davies_bouldin_score(pca_data, labels)
        if score < best_score:
            best_score = score
            best_k = k
    
    kmeans = KMeans(n_clusters=best_k, random_state=config.random_state, n_init=10)
    kmeans.fit(pca_data)
    cluster_labels = kmeans.labels_
    
    return FrozenPreprocessing(
        feature_names=feature_names,
        p99_thresholds={c: p99_dict[f"{c}_p99"] for c in feature_names},
        skewed_cols=skewed_cols,
        medians=medians,
        iqrs=iqrs,
        pca=pca,
        kmeans=kmeans,
        optimal_k=best_k,
    )


def assign_clusters_frozen(
    features_df: pl.DataFrame,
    frozen: FrozenPreprocessing
) -> np.ndarray:
    """Assign cluster labels using frozen preprocessing and model."""
    pca_data = frozen.transform(features_df)
    return frozen.kmeans.predict(pca_data)


def build_sankey_data(
    snapshots: list[SnapshotResult],
    config: MigrationConfig
) -> tuple[dict, pd.DataFrame]:
    """Build Sankey nodes, links, and switch-rate table from three snapshots.
    
    Returns:
        - Sankey dict with 'nodes', 'links', 'node_labels', 'node_colors'
        - Switch-rate DataFrame (only for customers in migration clusters at both adjacent snapshots)
    """
    # Get all unique customers across all snapshots
    all_customers = set()
    for snap in snapshots:
        all_customers.update(snap.features_df['customer_id'].to_list())
    all_customers = sorted(all_customers)
    
    # Build customer-state matrix: rows=customers, cols=snapshots
    # States: 'Not yet observed', 'No purchase in window', 'M1', 'M2', ...
    n_snapshots = len(snapshots)
    customer_states = pd.DataFrame(index=all_customers, columns=[f'Snap{i+1}' for i in range(n_snapshots)])
    customer_states = customer_states.fillna('Not yet observed')
    
    for i, snap in enumerate(snapshots):
        snap_customers = set(snap.features_df['customer_id'].to_list())
        # Customers with no purchases in this window but observed before
        for cust in all_customers:
            if cust in snap_customers:
                cluster = snap.cluster_labels[snap.features_df['customer_id'].to_list().index(cust)]
                customer_states.loc[cust, f'Snap{i+1}'] = f'M{cluster + 1}'
            elif cust in customer_states.index:
                # Check if customer existed before this snapshot
                prior_snaps = [snapshots[j].features_df['customer_id'].to_list() for j in range(i)]
                if any(cust in s for s in prior_snaps):
                    customer_states.loc[cust, f'Snap{i+1}'] = 'No purchase in window'
    
    # Build transition counts
    transitions = []
    for i in range(n_snapshots - 1):
        from_col = f'Snap{i+1}'
        to_col = f'Snap{i+2}'
        trans_counts = customer_states.groupby([from_col, to_col]).size().reset_index(name='count')
        transitions.append(trans_counts)
    
    # Combine all transitions
    all_trans = pd.concat(transitions, ignore_index=True)
    
    # Build Sankey nodes (distinct per column even if same label)
    # Format: "Snap1: M1", "Snap1: M2", ..., "Snap2: M1", ...
    node_labels = []
    node_colors = []
    node_to_idx = {}
    
    for i in range(n_snapshots):
        states_at_snap = customer_states[f'Snap{i+1}'].unique()
        # Order: Not yet observed, No purchase in window, then M1, M2, ...
        ordered = []
        if 'Not yet observed' in states_at_snap:
            ordered.append('Not yet observed')
        if 'No purchase in window' in states_at_snap:
            ordered.append('No purchase in window')
        m_states = sorted([s for s in states_at_snap if s.startswith('M')], key=lambda x: int(x[1:]))
        ordered.extend(m_states)
        
        for state in ordered:
            label = f'Snap{i+1}: {state}'
            node_to_idx[(i, state)] = len(node_labels)
            node_labels.append(label)
            
            # Color by state type
            if state == 'Not yet observed':
                node_colors.append('#94a3b8')  # slate
            elif state == 'No purchase in window':
                node_colors.append('#fbbf24')  # amber
            else:
                # Consistent colors for same M# across snapshots
                m_idx = int(state[1:]) - 1
                palette = ['#2563eb', '#ef4444', '#22c55e', '#a855f7', '#f97316', '#ec4899', '#06b6d4', '#84cc16']
                node_colors.append(palette[m_idx % len(palette)])
    
    # Build links
    links_source = []
    links_target = []
    links_value = []
    links_hover = []
    
    for i in range(n_snapshots - 1):
        from_col = f'Snap{i+1}'
        to_col = f'Snap{i+2}'
        trans_counts = customer_states.groupby([from_col, to_col]).size().reset_index(name='count')
        
        for _, row in trans_counts.iterrows():
            from_state = row[from_col]
            to_state = row[to_col]
            count = row['count']
            
            source_idx = node_to_idx[(i, from_state)]
            target_idx = node_to_idx[(i + 1, to_state)]
            
            links_source.append(source_idx)
            links_target.append(target_idx)
            links_value.append(int(count))
            
            # Hover: count and percentage of origin node
            origin_total = customer_states[from_col].value_counts().get(from_state, 1)
            pct = (count / origin_total) * 100
            links_hover.append(f'{from_state} → {to_state}: {count:,} customers ({pct:.1f}% of {from_state})')
    
    # Build switch-rate table (only M-to-M transitions at adjacent snapshots)
    switch_rows = []
    for i in range(n_snapshots - 1):
        from_col = f'Snap{i+1}'
        to_col = f'Snap{i+2}'
        m_from = customer_states[customer_states[from_col].str.startswith('M')]
        m_both = m_from[m_from[to_col].str.startswith('M')]
        
        if len(m_both) > 0:
            switch_rate = (m_both[from_col] != m_both[to_col]).mean() * 100
            total = len(m_both)
            from_counts = m_both.groupby(from_col)[to_col].apply(list).to_dict()
            switch_rows.append({
                'transition': f'{from_col} → {to_col}',
                'customers_in_migration_segments': total,
                'switch_rate_pct': round(switch_rate, 1),
            })
    
    switch_df = pd.DataFrame(switch_rows)
    
    sankey_data = {
        'node_labels': node_labels,
        'node_colors': node_colors,
        'links': {
            'source': links_source,
            'target': links_target,
            'value': links_value,
            'hover': links_hover,
        },
        'customer_states': customer_states,
    }
    
    return sankey_data, switch_df


def create_sankey_figure(sankey_data: dict) -> go.Figure:
    """Create Plotly Sankey figure from sankey_data dict."""
    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            label=sankey_data['node_labels'],
            color=sankey_data['node_colors'],
            pad=20,
            thickness=20,
            line=dict(color='rgba(0,0,0,0.1)', width=1),
        ),
        link=dict(
            source=sankey_data['links']['source'],
            target=sankey_data['links']['target'],
            value=sankey_data['links']['value'],
            color='rgba(100, 116, 139, 0.25)',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=sankey_data['links']['hover'],
        ),
    ))
    
    fig.update_layout(
        title='Customer Migration Across Three Snapshots',
        font=dict(family='Inter, system-ui, sans-serif', size=12, color='#1e293b'),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=40, r=40, t=60, b=40),
        height=600,
    )
    
    return fig


def build_snapshot_profiles(
    snapshots: list[SnapshotResult],
    config: MigrationConfig
) -> pd.DataFrame:
    """Build baseline migration-segment profiles for each snapshot."""
    rows = []
    
    for i, snap in enumerate(snapshots):
        # Get features with cluster labels
        features_with_cluster = snap.features_df.to_pandas()
        features_with_cluster['Migration_Segment'] = [f'M{c+1}' for c in snap.cluster_labels]
        
        # Numeric feature columns
        exclude_cols = {'customer_id', 'window_start', 'window_end', 'cutoff_date', 'Migration_Segment'}
        numeric_cols = [c for c in features_with_cluster.columns if c not in exclude_cols]
        
        for segment in sorted(features_with_cluster['Migration_Segment'].unique()):
            seg_data = features_with_cluster[features_with_cluster['Migration_Segment'] == segment]
            
            for col in numeric_cols:
                if col in seg_data.columns:
                    vals = seg_data[col].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(vals) > 0:
                        rows.append({
                            'snapshot': f'Snap{i+1}',
                            'cutoff_date': snap.cutoff_date,
                            'segment': segment,
                            'feature': col,
                            'mean': vals.mean(),
                            'median': vals.median(),
                            'std': vals.std(),
                            'count': len(vals),
                        })
    
    return pd.DataFrame(rows)


def render_segment_migration(
    tx_df: pl.DataFrame,
    source_fingerprint: str,
    key_prefix: str = 'segment_migration',
    config: Optional[MigrationConfig] = None,
) -> None:
    """Render the Segment Migration Sankey panel in Streamlit.
    
    Args:
        tx_df: Validated Polars DataFrame with transaction data
        source_fingerprint: SHA-256 fingerprint of the transaction source
        key_prefix: Prefix for Streamlit widget keys
        config: Optional MigrationConfig for customization
    
    The function manages its own session state and provides a build button
    to avoid re-running expensive feature engineering on every interaction.
    """
    import streamlit as st
    
    if config is None:
        config = MigrationConfig()
    
    st.subheader('Segment Migration (Three-Snapshot Sankey)')
    st.caption(
        'Tracks how customers move between migration segments across three historical snapshots. '
        'Segments M1, M2, … are defined by a model fitted on snapshot 1 and frozen for snapshots 2 and 3. '
        'They are not the same as the full-history clusters in the main dashboard.'
    )
    
    # Session state keys
    state_key = f'{key_prefix}_state'
    results_key = f'{key_prefix}_results'
    range_key = f'{key_prefix}_date_range'
    
    # Initialize session state
    if state_key not in st.session_state:
        st.session_state[state_key] = 'idle'
    if results_key not in st.session_state:
        st.session_state[results_key] = None
    if range_key not in st.session_state:
        st.session_state[range_key] = None
    
    # Get date range from transactions
    min_dt = tx_df.select(pl.col('transaction_date').min()).collect().item()
    max_dt = tx_df.select(pl.col('transaction_date').max()).collect().item()
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        range_start = st.date_input(
            'Range start',
            value=min_dt,
            min_value=min_dt,
            max_value=max_dt,
            key=f'{key_prefix}_range_start',
        )
    with col2:
        range_end = st.date_input(
            'Range end',
            value=max_dt,
            min_value=min_dt,
            max_value=max_dt,
            key=f'{key_prefix}_range_end',
        )
    
    # Validate range
    if range_start >= range_end:
        st.error('Range start must be before range end.')
        return
    
    # Check if range changed
    current_range = (range_start, range_end)
    if st.session_state[range_key] != current_range:
        st.session_state[range_key] = current_range
        st.session_state[state_key] = 'idle'
        st.session_state[results_key] = None
    
    # Build button
    if st.button('Build / refresh migration Sankey', key=f'{key_prefix}_build', type='primary'):
        st.session_state[state_key] = 'building'
        st.rerun()
    
    # Execute build if triggered
    if st.session_state[state_key] == 'building':
        with st.spinner('Building snapshots and fitting frozen model…'):
            try:
                # Select snapshot dates
                snap1_date, snap2_date, snap3_date = select_snapshot_dates(
                    tx_df, range_start, range_end
                )
                
                # Compute window days
                window_days = compute_window_days(range_start, range_end, config.max_window_days)
                
                # Build features for each snapshot
                snapshots = []
                for snap_date in [snap1_date, snap2_date, snap3_date]:
                    features = build_features_at_cutoff(tx_df, snap_date, window_days)
                    
                    if len(features) == 0:
                        raise ValueError(f'Snapshot {snap_date} has no customers with transactions in window.')
                    
                    snapshots.append(SnapshotResult(
                        cutoff_date=snap_date,
                        window_start=snap_date - timedelta(days=window_days),
                        window_end=snap_date,
                        features_df=features,
                        pca_data=None,  # Will fill below
                        cluster_labels=None,
                        n_customers=len(features),
                    ))
                
                # Check snapshot 1 has enough customers
                if snapshots[0].n_customers < config.min_customers_snapshot1:
                    raise ValueError(
                        f'Snapshot 1 has only {snapshots[0].n_customers} customers '
                        f'(minimum {config.min_customers_snapshot1}). '
                        'Try a wider date range or shorter window.'
                    )
                
                # Fit frozen model on snapshot 1
                frozen = fit_frozen_model(snapshots[0].features_df, config)
                
                # Assign clusters to all snapshots
                for i, snap in enumerate(snapshots):
                    if i == 0:
                        # Use the labels from fitting (already computed inside fit_frozen_model)
                        snap.cluster_labels = frozen.kmeans.labels_
                        snap.pca_data = frozen.pca.transform(
                            frozen.transform(snap.features_df)
                        )
                    else:
                        snap.cluster_labels = assign_clusters_frozen(snap.features_df, frozen)
                        snap.pca_data = frozen.transform(snap.features_df)
                
                # Build Sankey data
                sankey_data, switch_df = build_sankey_data(snapshots, config)
                
                # Build segment profiles
                profiles_df = build_snapshot_profiles(snapshots, config)
                
                # Store results
                st.session_state[results_key] = {
                    'snapshots': snapshots,
                    'frozen': frozen,
                    'sankey_data': sankey_data,
                    'switch_df': switch_df,
                    'profiles_df': profiles_df,
                    'window_days': window_days,
                    'snapshot_dates': [snap1_date, snap2_date, snap3_date],
                    'date_range': current_range,
                }
                st.session_state[state_key] = 'done'
                st.rerun()
                
            except Exception as e:
                st.error(f'Migration analysis failed: {e}')
                st.session_state[state_key] = 'error'
                st.session_state[results_key] = None
    
    # Render results
    results = st.session_state[results_key]
    if results is None:
        if st.session_state[state_key] == 'idle':
            st.info('Select a date range and click **Build / refresh migration Sankey** to generate the diagram.')
        return
    
    # Success: render outputs
    snapshots = results['snapshots']
    sankey_data = results['sankey_data']
    switch_df = results['switch_df']
    profiles_df = results['profiles_df']
    window_days = results['window_days']
    snapshot_dates = results['snapshot_dates']
    
    # Snapshot summary table
    st.markdown('### Snapshot windows and coverage')
    window_df = pd.DataFrame([
        {
            'Snapshot': f'Snap{i+1}',
            'Cutoff date': snap.cutoff_date,
            'Window start': snap.window_start,
            'Window end': snap.window_end,
            'Window (days)': window_days,
            'Customers with purchases': snap.n_customers,
        }
        for i, snap in enumerate(snapshots)
    ])
    st.dataframe(window_df, use_container_width=True, hide_index=True)
    
    # Sankey diagram
    st.markdown('### Migration Sankey')
    fig = create_sankey_figure(sankey_data)
    st.plotly_chart(fig, use_container_width=True, key=f'{key_prefix}_sankey')
    
    # Switch-rate table
    st.markdown('### Migration-segment switch rates')
    st.caption(
        'Only customers assigned to a migration segment (M1, M2, …) at both adjacent snapshots '
        'are counted. Activity-state transitions (e.g., M1 → No purchase in window) '
        'are shown in the Sankey but excluded from this table.'
    )
    if not switch_df.empty:
        st.dataframe(switch_df, use_container_width=True, hide_index=True)
    else:
        st.info('No customers were in migration segments at both adjacent snapshots.')
    
    # Baseline profiles
    st.markdown('### Baseline migration-segment profiles')
    st.caption(
        'Mean values per segment per snapshot. Features are computed from the equal-length '
        'window ending at each snapshot cutoff. Spend means window spend, not lifetime spend.'
    )
    if not profiles_df.empty:
        # Pivot for easier reading
        pivot = profiles_df.pivot_table(
            index=['snapshot', 'segment'],
            columns='feature',
            values='mean',
        ).reset_index()
        st.dataframe(pivot, use_container_width=True, hide_index=True)
    else:
        st.info('No profile data available.')
    
    # CSV export
    st.markdown('### Export')
    if not switch_df.empty:
        csv = switch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            'Download switch-rate table (CSV)',
            data=csv,
            file_name='segment_migration_switch_rates.csv',
            mime='text/csv',
            key=f'{key_prefix}_export_switch',
        )
    
    if not profiles_df.empty:
        csv = profiles_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            'Download segment profiles (CSV)',
            data=csv,
            file_name='segment_migration_profiles.csv',
            mime='text/csv',
            key=f'{key_prefix}_export_profiles',
        )
    
    # Sankey aggregate flows export
    sankey_links = pd.DataFrame({
        'source': sankey_data['links']['source'],
        'target': sankey_data['links']['target'],
        'value': sankey_data['links']['value'],
        'hover': sankey_data['links']['hover'],
    })
    # Map indices back to labels
    label_map = {i: label for i, label in enumerate(sankey_data['node_labels'])}
    sankey_links['source_label'] = sankey_links['source'].map(label_map)
    sankey_links['target_label'] = sankey_links['target'].map(label_map)
    
    csv = sankey_links.to_csv(index=False).encode('utf-8')
    st.download_button(
        'Download Sankey aggregate flows (CSV)',
        data=csv,
        file_name='segment_migration_sankey_flows.csv',
        mime='text/csv',
        key=f'{key_prefix}_export_sankey',
    )