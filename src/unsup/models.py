from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering



try:
    import umap
    UMAP = umap.UMAP
except Exception:  # pragma: no cover
    UMAP = None

@dataclass
class EmbeddingResult:
    Z: np.ndarray
    method: str
    params: Dict[str, Any]

def embed(X: np.ndarray, method: str = "pca", **kwargs) -> EmbeddingResult:
    method = method.lower()
    if method == "pca":
        model = PCA(n_components=2, **kwargs)
    elif method in {"tsne", "t-sne"}:
        model = TSNE(n_components=2, **kwargs)
    elif method == "umap":
        if UMAP is None:
            raise ImportError("umap-learn not installed")
        model = UMAP(n_components=2, **kwargs)
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    Z = model.fit_transform(X)
    return EmbeddingResult(Z=Z, method=method, params=kwargs)

def cluster(X: np.ndarray, method: str = "kmeans", **kwargs):
    """
    Normalizes parameter names across clustering methods and prevents
    irrelevant kwargs (like n_clusters) from leaking into methods that don't use them.
    """
    m = method.lower()

    # Pull a count parameter once, remove from kwargs so nothing leaks
    count = None
    if "n_components" in kwargs:
        count = int(kwargs.pop("n_components"))
    if "n_clusters" in kwargs:
        # prefer explicit n_components for GMM; otherwise allow n_clusters alias
        if count is None:
            count = int(kwargs.pop("n_clusters"))
        else:
            kwargs.pop("n_clusters")  # ensure it never leaks

    if m == "kmeans":
        n = count if count is not None else 3
        model = KMeans(n_clusters=n, **kwargs)
        labels = model.fit_predict(X)

    elif m == "dbscan":
        # DBSCAN ignores cluster count entirely
        model = DBSCAN(**kwargs)
        labels = model.fit_predict(X)

    elif m in {"agglo", "agglomerative", "hierarchical"}:
        n = count if count is not None else 2
        model = AgglomerativeClustering(n_clusters=n, **kwargs)
        labels = model.fit_predict(X)
    elif m in {"spectral"}:
        n = count if count is not None else 3
        model = SpectralClustering(
            n_clusters=n,
            assign_labels="kmeans",
            affinity="nearest_neighbors",
            **kwargs
        )
        labels = model.fit_predict(X)

    elif m in {"gmm", "gaussian_mixture"}:
        n = count if count is not None else 3
        model = GaussianMixture(n_components=n, **kwargs)
        model.fit(X)
        labels = model.predict(X)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels, model


