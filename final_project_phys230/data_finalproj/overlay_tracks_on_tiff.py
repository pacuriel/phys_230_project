# overlay_tracks_on_tiff.py
# Overlay simulation and experimental tracks on top of TIFF image stack

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tifffile import imread
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("tiff", type=Path, help="Path to TIFF stack")
parser.add_argument("--exp_csv", type=Path, help="CSV file with experimental tracks")
parser.add_argument("--sim_csv", type=Path, help="CSV file with simulated tracks")
parser.add_argument("--out", type=Path, default=Path("overlay.gif"))
parser.add_argument("--fps", type=int, default=5)
args = parser.parse_args()

# Load TIFF and determine dimensions
tiff = imread(str(args.tiff))

# Convert RGB to grayscale if needed
if tiff.ndim == 4 and tiff.shape[-1] == 3:
    tiff = np.mean(tiff, axis=-1)

# Ensure shape is (frames, H, W)
if tiff.ndim == 2:
    tiff = np.expand_dims(tiff, axis=0)
frames, H, W = tiff.shape


# Load track data
exp = pd.read_csv(args.exp_csv) if args.exp_csv else pd.DataFrame()
sim = pd.read_csv(args.sim_csv) if args.sim_csv else pd.DataFrame()

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
img = ax.imshow(tiff[0], cmap='gray')
exp_plot, = ax.plot([], [], 'ro', markersize=3, label='Exp')
sim_plot, = ax.plot([], [], 'bo', markersize=3, label='Sim')
ax.legend()

# Init function for animation
def init():
    exp_plot.set_data([], [])
    sim_plot.set_data([], [])
    return img, exp_plot, sim_plot

# Update function
def update(frame):
    img.set_data(tiff[frame])
    exp_pts = exp[exp['Frame'] == frame] if not exp.empty else pd.DataFrame()
    sim_pts = sim[sim['Frame'] == frame] if not sim.empty else pd.DataFrame()
    if not exp_pts.empty:
        exp_plot.set_data(exp_pts['X'], exp_pts['Y'])
    else:
        exp_plot.set_data([], [])
    if not sim_pts.empty:
        sim_plot.set_data(sim_pts['X'], sim_pts['Y'])
    else:
        sim_plot.set_data([], [])
    return img, exp_plot, sim_plot

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=1000 / args.fps)
ani.save(str(args.out), writer=PillowWriter(fps=args.fps))
print(f"[done] saved overlay to {args.out}")
