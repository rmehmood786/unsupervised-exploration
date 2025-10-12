#!/usr/bin/env python
import itertools, subprocess, os
from pathlib import Path

DATASETS = ["iris", "wine", "digits"]
EMBEDS = ["pca", "umap"]
CLUSTERS = ["kmeans", "gmm"]
N_CLUSTERS = {"iris": 3, "wine": 3, "digits": 10}

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    Path("outputs").mkdir(exist_ok=True)
    for d, e, c in itertools.product(DATASETS, EMBEDS, CLUSTERS):
        n = N_CLUSTERS[d]
        out = f"outputs/{d}_{e}_{c}"
        cmd = ["python", "scripts/run_experiment.py", "--dataset", d, "--embed", e, "--cluster", c, "--n-clusters", str(n), "--outdir", out]
        run(cmd)

if __name__ == "__main__":
    main()

    # --- Extra ablation sweeps (optional) ---
    # 1️⃣ Vary UMAP n_neighbors
    import subprocess
    for n in [5, 10, 15, 30, 50]:
        subprocess.run([
            "python", "scripts/run_experiment.py",
            "--dataset", "digits",
            "--embed", "umap",
            "--cluster", "gmm",
            "--n-clusters", "10",
            "--outdir", f"outputs/digits_umap{n}_gmm",
            "--n-neighbors", str(n)
        ], check=True)

    # 2️⃣ Vary UMAP min_dist
    for d in [0.0, 0.1, 0.5]:
        subprocess.run([
            "python", "scripts/run_experiment.py",
            "--dataset", "digits",
            "--embed", "umap",
            "--cluster", "kmeans",
            "--n-clusters", "10",
            "--outdir", f"outputs/digits_umap_min{d}_kmeans",
            "--min-dist", str(d)
        ], check=True)

    # 3️⃣ Vary GMM covariance type
    for cov in ["full", "tied", "diag", "spherical"]:
        subprocess.run([
            "python", "scripts/run_experiment.py",
            "--dataset", "digits",
            "--embed", "umap",
            "--cluster", "gmm",
            "--n-clusters", "10",
            "--outdir", f"outputs/digits_umap_gmm_{cov}",
            "--covariance-type", cov
        ], check=True)
