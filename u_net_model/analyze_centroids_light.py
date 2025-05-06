import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path

def analyze_centroids(csv_path):
    df = pd.read_csv(csv_path)
    summary = []

    for movie, group in df.groupby("Movie"):
        coords = group[['X', 'Y']].to_numpy()
        N = len(coords)
        print(f"\nAnalyzing {movie} with {N} cells")

        # Plot centroids
        plt.figure(figsize=(6, 6))
        plt.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.7)
        plt.title(f"Cell Centroids: {movie}")
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.savefig(f"{movie}_centroids.png")
        plt.close()

        # Nearest neighbor distances using KDTree
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)  # k=2 because first is self
        nn_dists = dists[:, 1]

        # Histogram
        plt.figure()
        plt.hist(nn_dists, bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Nearest Neighbor Distances: {movie}")
        plt.xlabel("Distance (pixels)")
        plt.ylabel("Frequency")
        plt.savefig(f"{movie}_nn_histogram.png")
        plt.close()

        # Density: cells per image area
        area = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
        density = N / area if area > 0 else 0

        summary.append({
            "Movie": movie,
            "NumCells": N,
            "MeanNN_Distance": nn_dists.mean(),
            "StdNN_Distance": nn_dists.std(),
            "CellDensity": density
        })

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("centroid_summary.csv", index=False)
    print("\nSaved: centroid_summary.csv, scatter plots, and histograms.")

# Run
if __name__ == "__main__":
    analyze_centroids("centroids.csv")
