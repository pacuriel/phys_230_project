# reorientations_hist.py
# Plot histogram of per-cell CIL-triggered reorientations from simulation

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def count_reorientations(sim_file):
    df = pd.read_csv(sim_file)
    if 'TrackID' not in df.columns or 'Reorient' not in df.columns:
        raise ValueError("Expected columns: 'TrackID' and 'Reorient'")
    return df.groupby('TrackID')['Reorient'].sum()

def plot_hist(counts, label, out):
    plt.figure(figsize=(6,4))
    plt.hist(counts, bins=30, alpha=0.7, label=label)
    plt.xlabel("Reorientations per Cell")
    plt.ylabel("Frequency")
    plt.title("Histogram of CIL-triggered Reorientation Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    print(f"[done] saved histogram: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Path to sim_tracks.csv")
    parser.add_argument("--out", type=Path, default="reorient_hist.png")
    parser.add_argument("--label", type=str, default="CIL ON")
    args = parser.parse_args()

    counts = count_reorientations(args.csv)
    plot_hist(counts, args.label, args.out)
