#!/usr/bin/env python
# collective_cil_emt_sim.py   v3  (animate OR save)

from __future__ import annotations
import argparse, math, random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import trange

# -------------------------------- util --------------------------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

# -------------------------------- model -------------------------------
@dataclass
class Cell:
    pos: np.ndarray
    pol: np.ndarray
    v0:  float
    R:   float
    epithelial: bool
    cil_timer: float = 0.0

@dataclass
class World:
    N:int; box:float=200.; dt:float=0.1
    v_mes:float=15.; v_epi:float=5.; R_mes:float=4.; R_epi:float=6.
    eps_rep:float=80.; eps_adh:float=10.; eta:float=0.2; tau_cil:float=3.
    p_emt:float=1e-3; p_met:float=5e-4
    cells:List[Cell]=field(init=False,default_factory=list)

    def __post_init__(self):
        while len(self.cells)<self.N:
            p=np.random.uniform(self.R_epi, self.box-self.R_epi,2)
            if all(np.linalg.norm(p-c.pos)>self.R_epi+c.R for c in self.cells):
                th=np.random.rand()*2*math.pi
                self.cells.append(Cell(
                    p, np.array([math.cos(th),math.sin(th)]),
                    self.v_epi, self.R_epi, True))

    def _force(self,a:Cell,b:Cell):
        r=b.pos-a.pos; d=np.linalg.norm(r)
        if d==0: return np.zeros(2)
        n=r/d
        rep=self.eps_rep*math.exp(-d/(a.R+b.R))
        adh=-self.eps_adh*math.exp(-d/((a.R+b.R)*1.4))
        return (rep+adh)*n

    def _cil(self,a,b):
        if np.linalg.norm(a.pos-b.pos)<a.R+b.R:
            a.pol=unit(a.pos-b.pos); b.pol=unit(b.pos-a.pos)
            a.cil_timer=b.cil_timer=self.tau_cil

    def step(self):
        # EMT / MET
        for c in self.cells:
            if c.epithelial and random.random()<self.p_emt*self.dt:
                c.epithelial=False; c.v0=self.v_mes; c.R=self.R_mes
            elif (not c.epithelial) and random.random()<self.p_met*self.dt:
                c.epithelial=True; c.v0=self.v_epi; c.R=self.R_epi
        # forces + cil
        forces=[np.zeros(2) for _ in self.cells]
        for i,a in enumerate(self.cells):
            for j in range(i+1,self.N):
                b=self.cells[j]
                F=self._force(a,b); forces[i]+=F; forces[j]-=F; self._cil(a,b)
        # update
        for idx,c in enumerate(self.cells):
            c.pol=unit(c.pol+self.eta*math.sqrt(self.dt)*np.random.randn(2))
            speed=c.v0*(0.3 if c.cil_timer>0 else 1.0)
            c.cil_timer=max(0,c.cil_timer-self.dt)
            c.pos += (speed*c.pol+forces[idx])*self.dt
            for k in (0,1):
                if c.pos[k]<c.R: c.pos[k]=c.R; c.pol[k]*=-1
                elif c.pos[k]>self.box-c.R: c.pos[k]=self.box-c.R; c.pol[k]*=-1

    # ---------------- run + optional frame capture ----------------
    def run(self, steps:int, capture_every:int|None=None):
        frames=[]
        for s in trange(steps, desc='sim'):
            if capture_every and s%capture_every==0:
                frames.append(np.array([c.pos.copy() for c in self.cells]))
            self.step()
        return frames

    def save_movie(self, frames:list[np.ndarray], out:Path, fps:int=20):
        fig,ax=plt.subplots(figsize=(6,6))
        ax.set_xlim(0,self.box); ax.set_ylim(0,self.box); ax.set_aspect('equal')
        scat=ax.scatter([],[],s=20)
        def init(): scat.set_offsets(np.empty((0,2))); return (scat,)
        def update(frame):
            scat.set_offsets(frame); return (scat,)
        ani=animation.FuncAnimation(
            fig, lambda i: (scat.set_offsets(frames[i]),)[0],
            init_func=init, frames=len(frames), interval=1000/fps, blit=True)
        ani.save(out, writer=animation.FFMpegWriter(fps=fps))
        plt.close(fig)

# ------------------------------ main ----------------------------------
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--N',type=int,default=100)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--dt',type=float,default=0.1)
    parser.add_argument('--animate',action='store_true')
    parser.add_argument('--save',type=Path,help='mp4 or gif')
    args=parser.parse_args()

    # backend logic
    if args.animate:
        try: matplotlib.use('TkAgg')
        except ImportError:
            print('[warn] GUI backend missing; falling back to --save sim.mp4')
            args.save=Path('sim.mp4'); args.animate=False
    else:
        matplotlib.use('Agg')

    sim=World(N=args.N, dt=args.dt)
    if args.animate:
        frames=sim.run(args.steps, capture_every=5)
        sim.save_movie(frames, Path('_tmp.mp4'), 20)  # write temp then show
        from subprocess import call; call(['xdg-open','_tmp.mp4'])
    elif args.save:
        frames=sim.run(args.steps, capture_every=5)
        sim.save_movie(frames, args.save, fps=20)
    else:
        sim.run(args.steps)  # headless, no output
