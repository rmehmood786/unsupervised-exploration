# Project Overview
This document accompanies the repo and explains the experimental pipeline:

1. Load a dataset (e.g., Iris) from `unsup.data`.
2. Compute a 2D embedding (PCA, t-SNE, UMAP) via `unsup.models.embed`.
3. Fit a clustering method on the embedding (KMeans/DBSCAN/Agglomerative/GMM).
4. Evaluate using internal scores and, when available, external scores.
5. Save plots and metrics to `outputs/`.

You can use this as a template for your own datasets.
