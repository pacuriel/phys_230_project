#!/usr/bin/env python
"""
tiff2tracks_multi.py
====================

Batch TIFF ➜ Cellpose segmentation ➜ Trackpy tracking ➜ CSV exporter

Example
-------
    python tiff2tracks_multi.py mid_1.tif mid_3.tif \
           --diameter 28 --model cyto --merge all_tracks.csv

Outputs
-------
tracks_out/
    tracks_mid_1.csv
    tracks_mid_3.csv
all_tracks.csv      (if --merge given)

Each CSV has columns: Movie, TrackID, Frame, T, X, Y
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from cellpose import models
import trackpy as tp


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def segment_stack(stack: np.ndarray, diameter: float, cp_model) -> pd.DataFrame:
    """
    Run Cellpose on every frame and return a table of centroids (Frame, X, Y).
    Compatible with Cellpose 2.x (returns 3-tuple) and 3.x (returns 4-tuple).
    """
    rows = []
    for frame, img in enumerate(stack):
        out = cp_model.eval(img, channels=[0, 0], diameter=diameter, progress=False)
        masks = out[0]  # first item is always the mask array
        for lab in np.unique(masks)[1:]:
            ys, xs = np.where(masks == lab)
            rows.append({"Frame": frame, "Y": float(ys.mean()), "X": float(xs.mean())})
    return pd.DataFrame(rows)


def link_centroids(df: pd.DataFrame, search_range: float) -> pd.DataFrame:
    """
    Link centroids over time with Trackpy.
    Trackpy expects lowercase 'x', 'y', 'frame' column names.
    """
    df = df.rename(columns={"X": "x", "Y": "y", "Frame": "frame"}).copy()
    df["particle"] = 0  # Trackpy overwrites this with TrackID
    linked = tp.link_df(df, search_range=search_range, memory=3)
    linked = linked.rename(
        columns={"particle": "TrackID", "frame": "Frame", "x": "X", "y": "Y"}
    )
    linked["T"] = linked["Frame"]  # assume Δt = 1 s placeholder
    return linked[["TrackID", "Frame", "T", "X", "Y"]]


def process_one_tiff(
    path: Path, diameter: float, model_type: str, out_dir: Path
) -> pd.DataFrame:
    """Segment & track a single TIFF, write per-movie CSV, return DataFrame."""
    stack = tiff.imread(path)
    if stack.ndim == 2:  # single frame → fake time axis
        stack = stack[None, ...]
    print(f"[info] {path.name}: {len(stack)} frames, size {stack.shape[1]}×{stack.shape[2]}")

    cp_model = models.Cellpose(model_type=model_type, gpu=False)
    spots = segment_stack(stack, diameter, cp_model)
    tracks = link_centroids(spots, search_range=diameter * 0.6)

    tracks["Movie"] = path.stem
    dest = out_dir / f"tracks_{path.stem}.csv"
    tracks.to_csv(dest, index=False)
    print(f"      → {dest}  ({tracks.TrackID.nunique()} tracks)")
    return tracks


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_cli(argv):
    p = argparse.ArgumentParser(description="Batch TIFF-to-tracks converter")
    p.add_argument("tiffs", nargs="+", type=Path, help="input .tif / .ome.tif stacks")
    p.add_argument("--diameter", "-d", type=float, required=True, help="cell diameter (pixels)")
    p.add_argument(
        "--model",
        "-m",
        choices=["cyto", "cyto2", "nuclei"],
        default="cyto",
        help="Cellpose model",
    )
    p.add_argument(
        "--outdir", "-o", type=Path, default=Path("tracks_out"),
        help="directory for per-movie CSVs",
    )
    p.add_argument(
        "--merge", "-M", type=Path,
        help="optional merged CSV filename (all movies)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_cli(argv or sys.argv[1:])
    args.outdir.mkdir(exist_ok=True)

    merged_tables = []
    for tif in args.tiffs:
        if not tif.is_file():
            print(f"[warn] {tif} not found, skipping")
            continue
        merged_tables.append(
            process_one_tiff(tif, args.diameter, args.model, args.outdir)
        )

    if args.merge and merged_tables:
        pd.concat(merged_tables, ignore_index=True).to_csv(args.merge, index=False)
        print(f"[done] merged file → {args.merge}")


if __name__ == "__main__":
    main()
