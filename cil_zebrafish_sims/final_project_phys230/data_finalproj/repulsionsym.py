from __future__ import annotations
import argparse, math, numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import List, Tuple

Vector = Tuple[float, float]
np.random.seed(42)

@dataclass
class Sphere:
    pos: np.ndarray  # (2,)
    vel: np.ndarray  # (2,)
    R: float = 1.0
    m: float = 1.0
    def move(self, dt: float):
        self.pos += self.vel * dt

class BilliardSim:
    def __init__(self, N:int=10, box:float=100.0, R:float=2.0, vmax:float=10.0):
        self.box=box
        self.spheres:List[Sphere]=[]
        tries=0
        while len(self.spheres)<N and tries<100*N:
            tries+=1
            p=np.random.uniform(R,box-R,size=2)
            if all(np.linalg.norm(p-s.pos)>2*R for s in self.spheres):
                v=np.random.uniform(-vmax,vmax,size=2)
                self.spheres.append(Sphere(p,v,R,1.0))
        if len(self.spheres)<N:
            raise RuntimeError("Could not place all spheres without overlap")
    # -------- physics helpers ---------
    @staticmethod
    def _resolve_pair(a:Sphere,b:Sphere):
        delta=b.pos-a.pos; dist=np.linalg.norm(delta)
        if dist==0: return
        n=delta/dist; rel=a.vel-b.vel; vn=np.dot(rel,n)
        if vn>0: return
        J=-2*vn/2
        a.vel+=J*n; b.vel-=J*n
        overlap=a.R+b.R-dist
        if overlap>0:
            a.pos-=n*overlap/2; b.pos+=n*overlap/2
    def _resolve_walls(self,s:Sphere):
        for k in (0,1):
            if s.pos[k]-s.R<0:
                s.pos[k]=s.R; s.vel[k]*=-1
            elif s.pos[k]+s.R>self.box:
                s.pos[k]=self.box-s.R; s.vel[k]*=-1
    # -------- step ---------
    def step(self,dt:float):
        for s in self.spheres: s.move(dt)
        for s in self.spheres: self._resolve_walls(s)
        for i in range(len(self.spheres)):
            for j in range(i+1,len(self.spheres)):
                if np.linalg.norm(self.spheres[i].pos-self.spheres[j].pos)<2*self.spheres[i].R:
                    self._resolve_pair(self.spheres[i],self.spheres[j])
    # -------- animation --------
    def animate(self,steps:int=2000,dt:float=0.1,interval:int=10):
        fig,ax=plt.subplots(figsize=(6,6))
        ax.set_xlim(0,self.box); ax.set_ylim(0,self.box); ax.set_aspect('equal')
        scat=ax.scatter([],[])
        def init():
            # Matplotlib ≥3.8 requires an (N,2) array; give it 0×2 instead of []
            import numpy as np
            scat.set_offsets(np.empty((0,2)))
            return scat,
        def update(_):
            self.step(dt)
            scat.set_offsets(np.array([s.pos for s in self.spheres]))
            return scat,
        ani=animation.FuncAnimation(fig,update,frames=steps,init_func=init,interval=interval,blit=True,repeat=False)
        plt.show()

def main():
    p=argparse.ArgumentParser();
    p.add_argument('--N',type=int,default=10); p.add_argument('--box',type=float,default=100.);
    p.add_argument('--R',type=float,default=2.0); p.add_argument('--steps',type=int,default=2000);
    p.add_argument('--dt',type=float,default=0.1); p.add_argument('--interval',type=int,default=10);
    args=p.parse_args();
    sim=BilliardSim(N=args.N,box=args.box,R=args.R)
    sim.animate(args.steps,args.dt,args.interval)
if __name__=='__main__':
    main()
