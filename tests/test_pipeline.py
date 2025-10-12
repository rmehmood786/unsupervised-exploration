from unsup.data import load_builtin
from unsup.models import embed, cluster
from unsup.evaluate import clustering_scores

def test_end_to_end():
    ds = load_builtin("iris")
    Z = embed(ds.X.values, method="pca").Z
    labels, _ = cluster(Z, method="kmeans", n_clusters=3)
    scores = clustering_scores(Z, labels, ds.y.values)
    assert "silhouette" in scores and "ari" in scores
