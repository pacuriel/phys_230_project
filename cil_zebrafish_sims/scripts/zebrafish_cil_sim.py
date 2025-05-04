# zebrafish_cil_sim.py (final with CIL toggle)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from pathlib import Path
import pandas as pd
from tqdm import trange

class CILSimulation:
    def __init__(self, positions, steps=1000, dt=1.0, diameter=10, box_size=600, cil_enabled=True):
        self.N = len(positions)
        self.steps = steps
        self.dt = dt
        self.diameter = diameter
        self.radius = diameter / 2
        self.box_size = box_size
        self.cil_enabled = cil_enabled

        self.positions = np.array(positions)
        angles = np.random.rand(self.N) * 2 * np.pi
        self.velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        self.history = []  # list of (frame, trackID, x, y, reorient)

    def step(self):
        new_positions = self.positions + self.velocities * self.dt
        reorient_flags = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            for j in range(i+1, self.N):
                delta = new_positions[i] - new_positions[j]
                dist = np.linalg.norm(delta)
                if dist < self.diameter and self.cil_enabled:
                    theta_i = np.random.rand() * 2 * np.pi
                    theta_j = np.random.rand() * 2 * np.pi
                    self.velocities[i] = [np.cos(theta_i), np.sin(theta_i)]
                    self.velocities[j] = [np.cos(theta_j), np.sin(theta_j)]
                    reorient_flags[i] = 1
                    reorient_flags[j] = 1

        for i in range(self.N):
            for d in range(2):
                if new_positions[i, d] < 0 or new_positions[i, d] > self.box_size:
                    self.velocities[i, d] *= -1
                    new_positions[i, d] = np.clip(new_positions[i, d], 0, self.box_size)

        self.positions = new_positions

        for i in range(self.N):
            self.history.append((self.current_step, i, *self.positions[i], reorient_flags[i]))

    def run(self):
        for self.current_step in trange(self.steps, desc="Simulating"):
            self.step()

    def save_tracks(self, out_path):
        df = pd.DataFrame(self.history, columns=["Frame", "TrackID", "X", "Y", "Reorient"])
        df.to_csv(out_path, index=False)
        print(f"[done] saved simulation tracks: {out_path}")

    def animate(self, out_path, fps=20):
        fig, ax = plt.subplots()
        scat = ax.scatter([], [], s=20)
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)

        def update(frame):
            pts = self.history[frame * self.N:(frame + 1) * self.N]
            coords = np.array([[x[2], x[3]] for x in pts])
            scat.set_offsets(coords)
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=self.steps, blit=True)
        ani.save(out_path, writer=animation.FFMpegWriter(fps=fps))
        print(f"[done] saved animation: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_csv", type=Path, help="CSV with initial X,Y positions")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--diameter", type=float, default=10)
    parser.add_argument("--save", type=str, help="Output mp4 animation filename")
    parser.add_argument("--tracks_out", type=str, default="sim_tracks.csv")
    parser.add_argument("--rhoa_knockdown", action="store_true", help="Disable CIL reorientation")
    args = parser.parse_args()

    if args.init_csv:
        df = pd.read_csv(args.init_csv)
        positions = df[['X', 'Y']].to_numpy()
    else:
        positions = np.random.rand(100, 2) * 600

    sim = CILSimulation(positions, steps=args.steps, dt=args.dt, diameter=args.diameter, cil_enabled=not args.rhoa_knockdown)
    sim.run()
    sim.save_tracks(args.tracks_out)

    if args.save:
        sim.animate(args.save)
