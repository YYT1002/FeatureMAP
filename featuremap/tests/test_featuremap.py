import numpy as np
import pytest
from sklearn.datasets import make_classification

from featuremap import FeatureMAP  # Assuming FeatureMAP is the class in your custom module


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
    featuremap = FeatureMAP(n_components=n_components, random_state=42)
    X_transformed = featuremap.fit_transform(X)

    # Check transformed data shape
    assert X_transformed.shape == (X.shape[0], n_components)

def test_featuremap_neighborhood_preservation(sample_data):
    """Test if FeatureMAP preserves local neighborhood structure."""
    X = sample_data
    n_components = 2

    # Fit FeatureMAP
    featuremap = FeatureMAP(n_components=n_components, random_state=42)
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
    featuremap1 = FeatureMAP(n_components=n_components, random_state=42)
    X_transformed1 = featuremap1.fit_transform(X)

    featuremap2 = FeatureMAP(n_components=n_components, random_state=42)
    X_transformed2 = featuremap2.fit_transform(X)

    # Check if results are identical
    assert np.allclose(X_transformed1, X_transformed2)



