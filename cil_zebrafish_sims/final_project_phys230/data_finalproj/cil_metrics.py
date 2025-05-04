# cil_metrics.py
"""
Contact‐Inhibition‑of‑Locomotion (CIL) Metrics
==============================================

Given a merged tracks CSV (columns: Movie, TrackID, Frame, T, X, Y) this script:

1. Detects contact events (distance < r_cil) between any pair of cells.
2. For each track, computes instantaneous speed and turning angle.
3. Extracts speed + Δθ windows before/after the first contact per track.
4. Saves:
   * `contact_events.csv` – one row per contact (Track_i, Track_j, Frame, dist)
   * `cil_metrics.pdf` – violin plot of speed drop & polar hist of turn angles.
   * `metrics_summary.txt` – median speed_pre, speed_post, median Δθ.

Usage
-----
    python cil_metrics.py all_tracks.csv --radius 15 --fps 3

Dependencies
------------
    pip install pandas numpy matplotlib seaborn tqdm
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def detect_contacts(df: pd.DataFrame, r: float) -> pd.DataFrame:
    events = []
    for frame, g in df.groupby('Frame'):
        coords = g[['X','Y']].values
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                dist = np.linalg.norm(coords[i]-coords[j])
                if dist < r:
                    events.append({'Frame': frame,
                                    'Track_i': g.iloc[i].TrackID,
                                    'Track_j': g.iloc[j].TrackID,
                                    'dist': dist})
    return pd.DataFrame(events)


def compute_kinematics(df: pd.DataFrame, fps: float):
    df = df.sort_values(['TrackID','Frame']).copy()
    df['dX'] = df.groupby('TrackID')['X'].diff()
    df['dY'] = df.groupby('TrackID')['Y'].diff()
    df['speed'] = np.sqrt(df.dX**2 + df.dY**2)*fps
    df['angle'] = np.arctan2(df.dY, df.dX)
    df['dtheta'] = df.groupby('TrackID')['angle'].diff()
    return df


def metrics_from_contacts(df: pd.DataFrame, contacts: pd.DataFrame):
    speeds_pre, speeds_post, dthetas = [], [], []
    first_contact = contacts.groupby('Track_i').Frame.min()
    for tid, contact_frame in first_contact.items():
        traj = df[df.TrackID==tid]
        pre = traj[traj.Frame==contact_frame-1]
        post = traj[traj.Frame==contact_frame+1]
        if not pre.empty and not post.empty:
            speeds_pre.append(pre.speed.values[0])
            speeds_post.append(post.speed.values[0])
            dthetas.append(abs(traj.loc[traj.Frame==contact_frame,'dtheta'].values[0]))
    return np.array(speeds_pre), np.array(speeds_post), np.array(dthetas)


def make_plots(s_pre, s_post, dθ, out_pdf):
    with plt.rc_context({'axes.titlesize':10}):
        fig, axes = plt.subplots(1,2, figsize=(8,4))
        sns.violinplot(data=[s_pre, s_post], ax=axes[0])
        axes[0].set_xticklabels(['pre','post']); axes[0].set_ylabel('Speed (µm/s)')
        axes[0].set_title('Speed drop after contact')
        axes[1].hist(dθ*180/np.pi, bins=18)
        axes[1].set_xlabel('Turn angle Δθ (deg)'); axes[1].set_title('Direction change')
        fig.tight_layout(); fig.savefig(out_pdf)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description='Compute CIL metrics from tracks CSV')
    p.add_argument('tracks', type=Path)
    p.add_argument('--radius', type=float, required=True, help='contact radius (µm)')
    p.add_argument('--fps', type=float, default=3, help='frames per second')
    p.add_argument('--outdir', type=Path, default=Path('cil_metrics'))
    args = p.parse_args(argv)
    args.outdir.mkdir(exist_ok=True)

    df = pd.read_csv(args.tracks)
    df = compute_kinematics(df, args.fps)

    print('[info] detecting contacts…')
    contacts = detect_contacts(df, args.radius)
    contacts.to_csv(args.outdir/'contact_events.csv', index=False)

    s_pre, s_post, dtheta = metrics_from_contacts(df, contacts)
    make_plots(s_pre, s_post, dtheta, args.outdir/'cil_metrics.pdf')

    with open(args.outdir/'metrics_summary.txt','w') as f:
        f.write(f"median speed pre-contact: {np.median(s_pre):.3f}\n")
        f.write(f"median speed post-contact: {np.median(s_post):.3f}\n")
        f.write(f"median |Δθ|: {np.median(dtheta*180/np.pi):.1f} deg\n")
    print('[done] results in', args.outdir)

if __name__ == '__main__':
    main()
