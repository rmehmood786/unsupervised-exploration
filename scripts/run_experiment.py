#!/usr/bin/env python
import argparse, json
import numpy as np
from pathlib import Path
from unsup.data import load_builtin
from unsup.models import embed, cluster
from unsup.evaluate import clustering_scores
from unsup.visualize import scatter2d
from unsup.evaluate import clustering_scores, best_label_permutation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["iris", "wine", "digits", "breast_cancer"])
    parser.add_argument("--embed", choices=["pca", "umap"], default="pca", help="Embedding method to apply before clustering")
    parser.add_argument("--cluster", choices=["kmeans", "dbscan", "agglo", "gmm", "spectral"])
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()


    ds = load_builtin(args.dataset)
    X = ds.X.values
    y = ds.y.values if ds.y is not None else None

    Z = embed(X, method=args.embed).Z
    labels, model = cluster(Z, method=args.cluster, n_clusters=args.n_clusters)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.outdir) / f"{args.dataset}_{args.embed}_{args.cluster}.png"
    scatter2d(Z, labels, title=f"{args.dataset} | {args.embed}+{args.cluster}", savepath=str(fig_path))

   
    scores = clustering_scores(Z, labels, y_true=y)
    if y is not None:
        _, mapping, acc, cm = best_label_permutation(y, labels)
        scores["cluster_label_accuracy"] = float(acc)
        # optional: save confusion matrix
        import json
        (Path(args.outdir) / "confusion_matrix.json").write_text(json.dumps(cm.tolist()))

    (Path(args.outdir) / "metrics.json").write_text(json.dumps(scores, indent=2))

    print("Saved figure to", fig_path)
    print("Scores:", json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()
