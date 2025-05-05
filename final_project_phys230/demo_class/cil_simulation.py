# Contact Inhibition of Locomotion (CIL) simulation

# This script simulates zebrafish endoderm cell behavior using an agent-based model with tunable parameters.
# It visualizes how cells interact, repel, or cluster depending on the state of contact inhibition.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from tqdm import tqdm

#PARAMETERS

# Core simulation parameters
N = 100                     # Number of cells
steps = 500                # Number of time steps
dt = 1.0                   # Time step size
box_size = 600             # Size of the square domain
cell_diameter = 20         # Diameter of each cell
speed = 2.0                # Speed of self-propulsion
cil_on = True             # Toggle for CIL behavior (True = on, False = off)
rhoa_knockdown = False    # Toggle to simulate loss of CIL (e.g., dominant-negative RhoA)

# Interaction strengths
repulsion_strength = 1.0   # Strength of repulsive force
cil_turn_rate = 0.5        # Rate at which direction is changed due to CIL

#INITIALIZATION

def initialize_positions(N, box_size, min_dist):
    positions = []
    while len(positions) < N:
        pos = np.random.rand(2) * box_size
        if all(np.linalg.norm(pos - p) > min_dist for p in positions):
            positions.append(pos)
    return np.array(positions)

def initialize_directions(N):
    angles = np.random.rand(N) * 2 * np.pi
    return np.column_stack((np.cos(angles), np.sin(angles)))

positions = initialize_positions(N, box_size, cell_diameter)
directions = initialize_directions(N)

#SIMULATION

positions_over_time = []

for t in tqdm(range(steps), desc="Simulating"):
    positions_over_time.append(positions.copy())
    
    # Compute repulsion and CIL
    new_directions = directions.copy()
    for i in range(N):
        force = np.zeros(2)
        for j in range(N):
            if i == j:
                continue
            d = positions[j] - positions[i]
            dist = np.linalg.norm(d)
            if dist < cell_diameter:
                # Repulsion
                force -= repulsion_strength * (d / dist)

                # CIL behavior
                if cil_on and not rhoa_knockdown:
                    away = -d / dist
                    new_directions[i] = (
                        (1 - cil_turn_rate) * new_directions[i] + cil_turn_rate * away
                    )
        # Normalize direction
        new_directions[i] /= np.linalg.norm(new_directions[i])
    
    directions = new_directions
    positions += speed * directions * dt

    # Reflecting boundaries
    positions = np.clip(positions, 0, box_size)

positions_over_time = np.array(positions_over_time)

#ANIMATION

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
scat = ax.scatter([], [], s=100, edgecolors='k')

def init():
    scat.set_offsets(np.empty((0, 2)))
    return scat,

def update(frame):
    scat.set_offsets(positions_over_time[frame])
    return scat,

ani = animation.FuncAnimation(
    fig, update, frames=steps, init_func=init, blit=True, interval=50
)

plt.title("CIL Simulation - {} cells".format(N))
plt.show()

ani.save("cil_demo.mp4", fps=20, writer='ffmpeg')


# To export the animation: ani.save('cil_demo.mp4', fps=20)
