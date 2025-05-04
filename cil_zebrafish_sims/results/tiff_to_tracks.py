# tiff_to_tracks.py
"""
TIFF → Segmentation → Tracking → CSV exporter
=============================================

Usage:
------
    python tiff_to_tracks.py movie.tif --diameter 30 --model cyto --out tracks.csv

Dependencies:
-------------
* tifffile
* cellpose  (pip install cellpose)
* trackpy   (pip install trackpy)
* pandas

What it does:
-------------
1. Loads a 2‑D time‑lapse stack from *movie.tif*.
2. Segments each frame with Cellpose (pre‑trained `--model`).
3. Extracts centroid coordinates of every mask.
4. Links centroids over time with `trackpy.link_df` (simple predictive tracker).
5. Writes a CSV with **TrackID, frame, t[s], x[µm], y[µm]** — ready for CIL analysis, ML features, or ABM calibration.

You can later swap Cellpose for your own U‑Net; just replace `segment_frame()`.
"""

from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from cellpose import models, utils
import trackpy as tp

# ------------------------------------------------------------
# Helper: segment a single frame with Cellpose and return masks
# ------------------------------------------------------------

def segment_frame(img: np.ndarray, model, diameter: float):
    masks, _, _ = model.eval(img, channels=[0, 0], diameter=diameter, progress=False)
    return masks

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def run(movie_path: Path, diameter: float, model_type: str, out_csv: Path):
    print(f"[info] loading stack {movie_path}…", flush=True)
    stack = tiff.imread(movie_path)
    if stack.ndim == 2:
        stack = stack[None, ...]  # treat single frame as T=1
    T, H, W = stack.shape
    print(f"[info] frames: {T}, size: {H}×{W}")

    model = models.Cellpose(model_type=model_type, gpu=False)
    rows = []
    start = time.time()
    for frame in range(T):
        masks = segment_frame(stack[frame], model, diameter)
        props = utils.regionprops(masks)
        for prop in props:
            y, x = prop.centroid
            rows.append({"frame": frame, "x": x, "y": y})
        if frame % 10 == 0:
            print(f"frame {frame}/{T-1} done", flush=True)
    seg_time = time.time() - start
    print(f"[info] segmentation finished in {seg_time:.1f}s — found {len(rows)} spots")

    df = pd.DataFrame(rows)
    df["particle"] = 0  # dummy placeholder for trackpy

    # link trajectories (predictive) — memory=3 frames gap allowed
    linked = tp.link_df(df, search_range=diameter, memory=3)
    linked = linked.rename(columns={"particle": "TrackID", "frame": "Frame", "x": "X", "y": "Y"})
    linked["T"] = linked["Frame"]  # assuming Δt = 1; edit if metadata gives time step

    linked.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv} with {linked['TrackID'].nunique()} tracks.")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description="Segment & track cells in a TIFF movie")
    p.add_argument("movie", type=Path, help="input .tif or .ome.tif stack")
    p.add_argument("--diameter", type=float, default=30, help="Cell diameter in pixels")
    p.add_argument("--model", choices=["cyto", "cyto2", "nuclei"], default="cyto", help="Cellpose model")
    p.add_argument("--out", type=Path, default=Path("tracks.csv"), help="output CSV path")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args.movie, args.diameter, args.model, args.out)
