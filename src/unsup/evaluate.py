from typing import Optional, Dict
import numpy as np
from sklearn import metrics
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, accuracy_score

def clustering_scores(X: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    scores = {
        "silhouette": metrics.silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan"),
        "calinski_harabasz": metrics.calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else float("nan"),
        "davies_bouldin": metrics.davies_bouldin_score(X, labels) if len(set(labels)) > 1 else float("nan"),
    }
    if y_true is not None:
        scores.update({
            "ari": metrics.adjusted_rand_score(y_true, labels),
            "nmi": metrics.normalized_mutual_info_score(y_true, labels),
            "homogeneity": metrics.homogeneity_score(y_true, labels),
            "completeness": metrics.completeness_score(y_true, labels),
            "v_measure": metrics.v_measure_score(y_true, labels),
        })
    return scores


def best_label_permutation(y_true: np.ndarray, y_pred: np.ndarray):
    # Map predicted cluster ids to true labels via Hungarian algorithm
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels_true)
    # cost = max - cm to maximize matches
    cost = cm.max() - cm
    r, c = linear_sum_assignment(cost)
    mapping = {labels_pred[j]: labels_true[i] for i, j in zip(r, c)}
    y_mapped = np.vectorize(lambda z: mapping.get(z, z))(y_pred)
    acc = accuracy_score(y_true, y_mapped)
    return y_mapped, mapping, acc, cm
