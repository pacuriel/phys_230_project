# analyze_sim_tracks.py
# Compute MSD and cluster size from simulation tracks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path
import argparse

def compute_msd(df, max_lag=50):
    msd = []
    for lag in range(1, max_lag+1):
        disps = []
        for tid, group in df.groupby('TrackID'):
            group = group.sort_values('Frame')
            pos = group[['X', 'Y']].to_numpy()
            if len(pos) > lag:
                diffs = pos[lag:] - pos[:-lag]
                disps.extend((diffs**2).sum(axis=1))
        msd.append(np.mean(disps))
    return np.array(msd)

def estimate_cluster_sizes(df, cutoff=20):
    cluster_sizes = []
    for frame, group in df.groupby('Frame'):
        pos = group[['X', 'Y']].to_numpy()
        if len(pos) == 0:
            continue
        tree = KDTree(pos)
        counts = tree.query_ball_tree(tree, r=cutoff)
        cluster_sizes.append(np.mean([len(c) for c in counts]))
    return np.array(cluster_sizes)

def main(sim_csv: Path, title: str):
    df = pd.read_csv(sim_csv)
    msd = compute_msd(df)
    clusters = estimate_cluster_sizes(df)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(msd, label=title)
    ax[0].set_title("Mean Squared Displacement")
    ax[0].set_xlabel("Lag time")
    ax[0].set_ylabel("MSD")

    ax[1].plot(clusters, label=title)
    ax[1].set_title("Mean Cluster Size")
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Avg. neighbors within radius")

    for a in ax: a.legend()
    plt.tight_layout()
    plt.savefig(f"analysis_{title}.png")
    print(f"[done] saved plot as analysis_{title}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_csv", type=Path, help="Simulation track CSV")
    parser.add_argument("--title", type=str, default="sim")
    args = parser.parse_args()
    main(args.sim_csv, args.title)
