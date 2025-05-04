# compare_reorient_hist.py
# Usage: python compare_reorient_hist.py --on sim_tracks.csv --off sim_tracks_rhoa_dn.csv --out reorient_hist_compare.png

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--on", type=Path, required=True, help="Path to CIL ON sim_tracks.csv")
parser.add_argument("--off", type=Path, required=True, help="Path to CIL OFF sim_tracks_rhoa_dn.csv")
parser.add_argument("--out", type=Path, default="reorient_hist_compare.png", help="Output plot filename")
args = parser.parse_args()

# Load and compute reorientation counts
on = pd.read_csv(args.on)
off = pd.read_csv(args.off)
on_counts = on.groupby("TrackID")["Reorient"].sum()
off_counts = off.groupby("TrackID")["Reorient"].sum()

# Plot
plt.figure(figsize=(8, 5))
plt.hist(on_counts, bins=30, alpha=0.7, label="CIL ON")
plt.hist(off_counts, bins=30, alpha=0.7, label="CIL OFF (RhoA-DN)")
plt.xlabel("Reorientations per Cell")
plt.ylabel("Frequency")
plt.title("Reorientation Histogram: CIL ON vs OFF")
plt.legend()
plt.tight_layout()
plt.savefig(args.out)
print(f"[done] saved histogram comparison: {args.out}")
