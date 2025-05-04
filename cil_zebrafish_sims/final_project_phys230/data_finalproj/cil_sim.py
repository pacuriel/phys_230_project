# cil_sim.py
"""
Minimal Agent‑Based Contact‑Inhibition‑of‑Locomotion (CIL) Simulator
===================================================================

This extends *sphere_repulsion_sim.py* into an active‑matter model that
mimics epithelial➜mesenchymal transitions with CIL‑style re‑polarisation.

Key additions
-------------
* **Self‑propulsion** – each agent has an intrinsic polarity vector `p` and
  moves with speed `v0` in that direction (Active Brownian particle).
* **CIL rule** – whenever two agents overlap, both reverse (or randomise)
  their polarity and optionally reduce speed for a refractory period `τ_cil`.
* **EMT switch** – agents carry a boolean `epithelial` flag:
    * `epithelial=True` → higher adhesion radius, lower motility.
    * `epithelial=False` (mesenchymal) → faster, weaker adhesion.  A user‑
      set fraction of cells switches phenotype after `t_emt` seconds.
* **Soft repulsion** – Morse‑type potential prevents unreal overlap while
  allowing small compression.

Run:
-----
    python cil_sim.py --N 50 --emt_frac 0.3 --steps 10000

Dependencies
------------
* numpy
* matplotlib>=3.8

Planned: hook this into the ML/ABM pipeline by logging `positions.npy`
(for training) and `cil_events.csv`.
"""
from __future__ import annotations
import argparse, math, numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from dataclasses import dataclass, field

np.random.seed(0)

@dataclass
class Cell:
    pos: np.ndarray         # (2,)
    pol: np.ndarray         # unit vector
    v0:  float
    R:   float
    epithelial: bool = True
    cil_timer: float = 0.0  # time left in refractory

@dataclass
class CILSim:
    N: int = 30
    box: float = 200.
    R_epi: float = 6.
    R_mes: float = 4.
    v_epi: float = 5.
    v_mes: float = 15.
    eta: float = 0.2       # rotational noise
    tau_cil: float = 3.0   # sec of slowed motility after contact
    dt: float = 0.1
    emt_frac: float = 0.3
    t_emt: float = 50.0

    cells: list[Cell] = field(init=False, default_factory=list)
    t: float = 0.0

    def __post_init__(self):
        # init positions without overlap
        while len(self.cells) < self.N:
            pos = np.random.uniform(self.R_epi, self.box - self.R_epi, size=2)
            if all(np.linalg.norm(pos - c.pos) > (self.R_epi + self.R_epi) for c in self.cells):
                theta = np.random.rand() * 2 * np.pi
                self.cells.append(Cell(pos, np.array([math.cos(theta), math.sin(theta)]),
                                        self.v_epi, self.R_epi, True))

    # soft Morse repulsion/adhesion
    def _force(self, a: Cell, b: Cell):
        r = b.pos - a.pos; dist = np.linalg.norm(r)
        if dist == 0: return np.zeros(2)
        n = r / dist
        # adhesion radius larger for epithelial
        Radh = (a.R + b.R) * (1.4 if a.epithelial or b.epithelial else 1.1)
        eps = 10.
        if dist < a.R + b.R:  # strong repulsion
            return 200 * n
        elif dist < Radh:     # weak attraction
            return -eps * n * (1 - dist / Radh)
        return np.zeros(2)

    def _cil_contact(self, a: Cell, b: Cell):
        if np.linalg.norm(a.pos - b.pos) < a.R + b.R:
            # flip their polarities and start refractory period
            for c in (a, b):
                c.pol *= -1
                c.cil_timer = self.tau_cil

    def step(self):
        self.t += self.dt
        # EMT conversion
        if self.t > self.t_emt:
            to_switch = np.random.choice(self.N, int(self.emt_frac * self.N), replace=False)
            for i in to_switch:
                if self.cells[i].epithelial:
                    self.cells[i].epithelial = False
                    self.cells[i].R = self.R_mes
                    self.cells[i].v0 = self.v_mes

        # forces & contacts
        forces = [np.zeros(2) for _ in self.cells]
        for i, a in enumerate(self.cells):
            for j in range(i + 1, self.N):
                b = self.cells[j]
                F = self._force(a, b)
                forces[i] += F
                forces[j] -= F
                self._cil_contact(a, b)
        # update cells
        for i, c in enumerate(self.cells):
            # rotational noise
            c.pol += self.eta * np.random.randn(2)
            c.pol /= np.linalg.norm(c.pol)
            speed = 0.3 * c.v0 if c.cil_timer > 0 else c.v0
            c.cil_timer = max(0, c.cil_timer - self.dt)
            vel = speed * c.pol + forces[i]
            c.pos += vel * self.dt
            # walls
            for k in (0,1):
                if c.pos[k] < c.R:
                    c.pos[k] = c.R; c.pol[k] *= -1
                elif c.pos[k] > self.box - c.R:
                    c.pos[k] = self.box - c.R; c.pol[k] *= -1

    # ---------- visualisation ---------- #
    def animate(self, steps=2000, interval=20):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0, self.box); ax.set_ylim(0, self.box); ax.set_aspect('equal')
        scat = ax.scatter([],[] , s=25)
        def init():
            scat.set_offsets(np.empty((0,2)))
            return scat,
        def update(_):
            self.step()
            xy = np.array([c.pos for c in self.cells])
            scat.set_offsets(xy)
            colors = ['#1f77b4' if c.epithelial else '#ff7f0e' for c in self.cells]
            scat.set_color(colors)
            return scat,
        ani = anim.FuncAnimation(fig, update, init_func=init, frames=steps,
                                  interval=interval, blit=True, repeat=False)
        plt.show()

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=30)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--interval', type=int, default=20)
    p.add_argument('--emt_frac', type=float, default=0.3)
    return p.parse_args()

if __name__ == '__main__':
    a = parse(); sim = CILSim(N=a.N, emt_frac=a.emt_frac)
    sim.animate(a.steps, a.interval)
