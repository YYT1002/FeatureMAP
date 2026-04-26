import numpy as np
import pytest
from sklearn.datasets import make_classification

from featuremap import FeatureMAP  # Assuming FeatureMAP is the class in your custom module
from featuremap.featuremap_ import (
    _compute_log_trace_elbow_step,
    _evaluate_online_log_delta_elbow_stop,
    _graph_convolution_dense_legacy,
    _prepare_graph_convolution_batches,
    graph_convolution,
)


try:
    # works for sklearn>=0.22
    from sklearn.manifold import trustworthiness
except ImportError:
    # this is to comply with requirements (scikit-learn>=0.20)
    # More recent versions of sklearn have exposed trustworthiness
    # in top level module API
    # see: https://github.com/scikit-learn/scikit-learn/pull/15337
    from sklearn.manifold.t_sne import trustworthiness


@pytest.fixture
def sample_data():
    """Generate a toy dataset for testing."""
    X, _ = make_classification(
        n_samples=20,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    return X

def test_featuremap_transformed_shape(sample_data):
    """Test if FeatureMAP reduces the dimensionality correctly."""
    X = sample_data
    n_components = 2

    # Using FeatureMAP
    featuremap = FeatureMAP(n_components=n_components, n_neighbors=5, random_state=42)
    X_transformed = featuremap.fit_transform(X)

    # Check transformed data shape
    assert X_transformed.shape == (X.shape[0], n_components)

def test_featuremap_neighborhood_preservation(sample_data):
    """Test if FeatureMAP preserves local neighborhood structure."""
    X = sample_data
    n_components = 2

    # Fit FeatureMAP
    featuremap = FeatureMAP(n_components=n_components, n_neighbors=5, random_state=42)
    X_transformed = featuremap.fit_transform(X)

    # Compute pairwise distances in original and transformed space
    from sklearn.metrics import pairwise_distances
    original_distances = pairwise_distances(X)
    transformed_distances = pairwise_distances(X_transformed)

    # Check if nearest neighbors are preserved (correlation of distances)
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(original_distances.ravel(), transformed_distances.ravel())

    # FeatureMAP should preserve local structure (higher correlation for nearby points)
    assert correlation > 0.5  # Adjust threshold as needed

def test_featuremap_reproducibility(sample_data):
    """Test if FeatureMAP produces the same results with the same random state."""
    X = sample_data
    n_components = 2

    # Fit FeatureMAP with the same random state
    featuremap1 = FeatureMAP(n_components=n_components, n_neighbors=5, random_state=42)
    X_transformed1 = featuremap1.fit_transform(X)

    featuremap2 = FeatureMAP(n_components=n_components, n_neighbors=5, random_state=42)
    X_transformed2 = featuremap2.fit_transform(X)

    # Check if results are identical
    assert np.allclose(X_transformed1, X_transformed2)


def test_compute_log_trace_elbow_step_detects_fast_drop_then_tail():
    step_ids = np.arange(1, 9, dtype=np.int32)
    delta = np.array([1.0, 0.5, 0.25, 0.2, 0.18, 0.17, 0.165, 0.16], dtype=np.float64)

    elbow_step = _compute_log_trace_elbow_step(step_ids, delta)

    assert elbow_step == 3


def test_compute_log_trace_elbow_step_returns_current_step_for_flat_trace():
    step_ids = np.arange(1, 6, dtype=np.int32)
    delta = np.ones(step_ids.shape[0], dtype=np.float64)

    elbow_step = _compute_log_trace_elbow_step(step_ids, delta)

    assert elbow_step == int(step_ids[-1])


def test_compute_log_trace_elbow_step_handles_very_small_values():
    step_ids = np.arange(1, 5, dtype=np.int32)
    delta = np.array([1e-4, 1e-8, 1e-12, 1e-16], dtype=np.float64)

    elbow_step = _compute_log_trace_elbow_step(step_ids, delta)

    assert elbow_step == 3


def test_online_log_delta_elbow_stop_respects_min_steps_and_lookahead():
    delta = np.array([1.0, 0.5, 0.25, 0.2, 0.18, 0.17, 0.165, 0.16], dtype=np.float64)
    elbow_history = []
    stop_step = None

    for step in range(1, delta.shape[0] + 1):
        elbow_step, candidate_step, stable, eligible = _evaluate_online_log_delta_elbow_stop(
            np.arange(1, step + 1, dtype=np.int32),
            delta[:step],
            elbow_history,
            min_steps=4,
            stability=3,
            tolerance_steps=0,
            lookahead=2,
        )
        if step < 4:
            assert (elbow_step, candidate_step, stable, eligible) == (-1, -1, False, False)
        if eligible and stop_step is None:
            stop_step = step

    assert stop_step == 6


def test_featuremap_auto_delta_patience_stops_on_delta_only(sample_data):
    featuremap = FeatureMAP(
        n_components=2,
        n_neighbors=5,
        random_state=42,
        output_variation=True,
        collect_variation_pc_steps=True,
        gcn_stop_mode="auto_delta_patience",
        gcn_stop_delta_tol=10.0,
        gcn_stop_patience=2,
        gcn_max_iterations=4,
    )
    featuremap.fit_transform(sample_data)

    kwds = featuremap._featuremap_kwds

    assert kwds["gcn_stop_mode"] == "auto_delta_patience"
    assert int(kwds["gcn_chosen_iteration"]) == 2
    assert kwds["variation_pc_steps"].shape[0] == 3
    assert "gcn_stop_rw_residual" not in kwds


def test_featuremap_gcn_iterations_runs_exact_steps(sample_data):
    featuremap = FeatureMAP(
        n_components=2,
        n_neighbors=5,
        random_state=42,
        output_variation=True,
        collect_variation_pc_steps=True,
        gcn_iterations=4,
        gcn_stop_mode="auto_delta_patience",
        gcn_stop_delta_tol=10.0,
        gcn_stop_patience=1,
    )
    featuremap.fit_transform(sample_data)

    kwds = featuremap._featuremap_kwds

    assert int(kwds["gcn_chosen_iteration"]) == 4
    assert kwds["gcn_stop_reason"] == "fixed_iterations"
    assert int(kwds["gcn_iterations_used"]) == 4
    assert kwds["gcn_max_iterations_used"] is None
    assert kwds["variation_pc_steps"].shape[0] == 5
    assert np.any(kwds["gcn_stop_eligible"])


def test_featuremap_gcn_iterations_overrides_gcn_max_iterations_with_warning(sample_data):
    featuremap = FeatureMAP(
        n_components=2,
        n_neighbors=5,
        random_state=42,
        output_variation=True,
        collect_variation_pc_steps=True,
        gcn_iterations=3,
        gcn_max_iterations=2,
    )
    with pytest.warns(UserWarning, match="gcn_max_iterations"):
        featuremap.fit_transform(sample_data)

    kwds = featuremap._featuremap_kwds

    assert int(kwds["gcn_chosen_iteration"]) == 3
    assert kwds["gcn_stop_reason"] == "fixed_iterations"
    assert int(kwds["gcn_iterations_used"]) == 3


def test_featuremap_auto_elbow_log_delta_collects_consistent_variation_steps(sample_data):
    featuremap = FeatureMAP(
        n_components=2,
        n_neighbors=5,
        random_state=42,
        output_variation=True,
        collect_variation_pc_steps=True,
        gcn_stop_mode="auto_elbow_log_delta",
        gcn_max_iterations=4,
    )
    featuremap.fit_transform(sample_data)

    kwds = featuremap._featuremap_kwds

    assert kwds["variation_pc_steps"].shape[0] == kwds["gcn_chosen_iteration"] + 1
    assert np.allclose(
        kwds["variation_pc_steps"][-1],
        kwds["variation_pc"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_featuremap_k_dim_gex_embedding_shape():
    """k-dim gex embedding should produce VH_embedding of shape (n, k, k)
    and run the gex optimization without indexing errors when n_components > 2."""
    X, _ = make_classification(
        n_samples=120,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        random_state=0,
    )
    k = 3
    featuremap = FeatureMAP(
        n_components=k,
        n_neighbors=10,
        random_state=42,
        feat_lambda=0.1,
        feat_frac=0.2,
        gcn_max_iterations=2,
        threshold=0.99,
    )
    X_transformed = featuremap.fit_transform(X)

    assert X_transformed.shape == (X.shape[0], k)

    kwds = featuremap._featuremap_kwds
    assert kwds["VH_embedding"].shape == (X.shape[0], k, k)
    # Each frame row should be unit-normalized.
    row_norms = np.linalg.norm(kwds["VH_embedding"], axis=2)
    assert np.allclose(row_norms, 1.0, atol=1e-5)
    # Per-PC embeddings should be exposed under both legacy and indexed keys.
    assert kwds["gauge_v1_emb"].shape == (X.shape[0], k)
    assert kwds["gauge_v2_emb"].shape == (X.shape[0], k)
    assert kwds["gauge_v3_emb"].shape == (X.shape[0], k)


def test_featuremap_default_gcn_align_top_k_tracks_n_components(sample_data):
    featuremap = FeatureMAP(
        n_components=3,
        n_neighbors=5,
        random_state=42,
        output_variation=True,
        gcn_max_iterations=2,
    )
    featuremap.fit_transform(sample_data)

    kwds = featuremap._featuremap_kwds

    assert int(kwds["gcn_align_top_k"]) == min(3, int(kwds["VH"].shape[1]))


def test_featuremap_explicit_gcn_align_top_k_override_wins(sample_data):
    featuremap = FeatureMAP(
        n_components=3,
        n_neighbors=5,
        random_state=42,
        output_variation=True,
        gcn_align_top_k=1,
        gcn_max_iterations=2,
    )
    featuremap.fit_transform(sample_data)

    kwds = featuremap._featuremap_kwds

    assert int(kwds["gcn_align_top_k"]) == min(1, int(kwds["VH"].shape[1]))


def test_graph_convolution_fast_aligned_matches_legacy_small_case():
    rng = np.random.default_rng(0)
    n_nodes, f_dim, c_dim, k = 16, 6, 10, 5
    features = rng.normal(size=(n_nodes, f_dim, c_dim)).astype(np.float32)
    norms = np.linalg.norm(features, axis=2, keepdims=True)
    features /= np.maximum(norms, 1e-12)
    knn_index = ((np.arange(n_nodes)[:, None] + np.arange(k)[None, :]) % n_nodes).astype(np.int32)
    batch_metadata = _prepare_graph_convolution_batches(knn_index, batch_size=8)

    legacy_features, _ = _graph_convolution_dense_legacy(
        features.copy(),
        batch_metadata,
        num_iterations=2,
        align_top_k=2,
        eps=1e-12,
    )
    fast_features, _ = graph_convolution(
        features.copy(),
        knn_index,
        num_iterations=2,
        align_top_k=2,
        backend="auto",
        verbose=False,
    )

    assert np.allclose(fast_features, legacy_features, atol=1e-5, rtol=1e-5)


def test_graph_convolution_fast_aligned_tail_matches_legacy():
    rng = np.random.default_rng(1)
    n_nodes, f_dim, c_dim, k = 20, 7, 8, 6
    features = rng.normal(size=(n_nodes, f_dim, c_dim)).astype(np.float32)
    norms = np.linalg.norm(features, axis=2, keepdims=True)
    features /= np.maximum(norms, 1e-12)
    knn_index = ((np.arange(n_nodes)[:, None] + np.arange(k)[None, :]) % n_nodes).astype(np.int32)
    batch_metadata = _prepare_graph_convolution_batches(knn_index, batch_size=10)

    legacy_features, _ = _graph_convolution_dense_legacy(
        features.copy(),
        batch_metadata,
        num_iterations=2,
        align_top_k=2,
        eps=1e-12,
    )
    fast_features, _ = graph_convolution(
        features.copy(),
        knn_index,
        num_iterations=2,
        align_top_k=2,
        backend="auto",
        verbose=False,
    )

    assert np.allclose(fast_features[:, 2:, :], legacy_features[:, 2:, :], atol=1e-6, rtol=1e-6)


def test_graph_convolution_fallback_align_gt_4_matches_legacy():
    rng = np.random.default_rng(2)
    n_nodes, f_dim, c_dim, k = 18, 7, 9, 5
    features = rng.normal(size=(n_nodes, f_dim, c_dim)).astype(np.float32)
    norms = np.linalg.norm(features, axis=2, keepdims=True)
    features /= np.maximum(norms, 1e-12)
    knn_index = ((np.arange(n_nodes)[:, None] + np.arange(k)[None, :]) % n_nodes).astype(np.int32)
    batch_metadata = _prepare_graph_convolution_batches(knn_index, batch_size=9)

    legacy_features, _ = _graph_convolution_dense_legacy(
        features.copy(),
        batch_metadata,
        num_iterations=2,
        align_top_k=5,
        eps=1e-12,
    )
    fallback_features, _ = graph_convolution(
        features.copy(),
        knn_index,
        num_iterations=2,
        align_top_k=5,
        backend="auto",
        verbose=False,
    )

    assert np.allclose(fallback_features, legacy_features, atol=1e-6, rtol=1e-6)
