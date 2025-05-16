"""
File to simulate MET process using random walk model. 
Pablo Curiel
May 2025
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class SimMET:
    """Class to simulate MET process"""
    def __init__(self,
                 N: int,
                 num_steps: int,
                 step_size: float,
                 sim_met: bool = True,
                 interaction_radius: float = 2,
                 bbox_size: int = 10,
                 show_paths: bool = True,
                 save_sim_data: bool = True,
                 save_sim_fig: bool = True) -> None:
        """
        SimMet class constructor to set up simulation.

        Input:
            N: number of cells
            num_steps: number of time steps to simulate
            step_size: step cells should take in each direction
            bbox_size: bounding box size of figure to show of simulation
        """
        # Sim parameters
        self.N = N
        self.num_steps = num_steps
        self.step_size = step_size
        self.sim_met = sim_met
        self.save_sim_data = save_sim_data
        self.save_sim_fig = save_sim_fig
        
        self.seed = np.random.randint(0, 2**10) # Simulation seed
        np.random.seed(seed=self.seed)

        # MET parameters
        if self.sim_met:
            self.interaction_radius = interaction_radius # Radius for cells to interact

        self.bbox_size = bbox_size
        self.show_paths = show_paths

        self.set_sim_vars()

    def gen_sim_name(self):
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.sim_name = f"sim_N{self.N}_steps{self.num_steps}_step{self.step_size}_met{self.sim_met}_seed{self.seed}_{time}"

    def set_sim_vars(self):
        """Set variables relevant to simulations."""
        self.gen_sim_name()
        self.positions = np.random.uniform(-self.bbox_size // 2, self.bbox_size // 2, size=(self.N, 2))
        self.angles = np.random.rand(self.N) * 2 * np.pi
        self.trajectories = np.zeros((self.num_steps, self.N, 2))  # to store trails

        self.noise = 1.2 # Noise for increased randomness
        
        self.setup_plotting() # Settting plotting variables
        
    def setup_plotting(self):
        """Setting up plotting variables."""
        # Plotting variables
        self.fig, self.ax = plt.subplots()
        # self.bbox_size = 10 # Plot size 
        self.ax.set_xlim(-self.bbox_size, self.bbox_size)
        self.ax.set_ylim(-self.bbox_size, self.bbox_size)

        self.colors = plt.cm.jet(np.linspace(0, 1, self.N))  # Unique color for each cell
        
        self.scatters = [self.ax.plot([], [], 'o', color=self.colors[i])[0] for i in range(self.N)]
        
        # self.show_paths = True # Flag whether to show the cell paths
        if self.show_paths:
            self.trails = [self.ax.plot([], [], '-', color=self.colors[i], alpha=0.5)[0] for i in range(self.N)]

    def get_new_angles(self) -> np.ndarray:
        """Obtains new direction angles for cells."""
        new_angles = np.zeros(self.N)

        # Loop over all cells
        for i in range(self.N):
            # Get distance from current cell (i) to all others
            dx = self.positions[:, 0] - self.positions[i, 0]
            dy = self.positions[:, 1] - self.positions[i, 1]
            dists = np.sqrt(dx**2 + dy**2)

            # Get indices of neighbors (including self)
            neighbor_idx = dists < self.interaction_radius # Indices of cells within IR

            if neighbor_idx.sum() > 1:
                neighbor_angles = self.angles[neighbor_idx] # Angles of cells within IR

                # Compute mean angle using vector averaging
                mean_cos = np.mean(np.cos(neighbor_angles))
                mean_sin = np.mean(np.sin(neighbor_angles))
                avg_angle = np.arctan2(mean_sin, mean_cos)

                new_angles[i] = avg_angle
            else: # No cells within radius
                new_angles[i] = np.random.rand() * 2 * np.pi # Getting random angles
        
        new_angles += np.random.uniform(-self.noise, self.noise, size=self.N) # Adding movement noise
        
        return new_angles 

    def apply_bounds(self) -> None:
        """Bounds cells inside of plot region by bouncing off walls."""
        # Loop over cell
        for i in range(self.N):
            # Loop over both (x,y) coords
            for dim in range(2):  # 0 = x, 1 = y
                if self.positions[i, dim] > self.bbox_size:
                    self.positions[i, dim] = 2 * self.bbox_size - self.positions[i, dim]
                    self.angles[i] = np.pi - self.angles[i] if dim == 0 else -self.angles[i]
                elif self.positions[i, dim] < -self.bbox_size:
                    self.positions[i, dim] = -2 * self.bbox_size - self.positions[i, dim]
                    self.angles[i] = np.pi - self.angles[i] if dim == 0 else -self.angles[i]

    def update_met_state(self, t: int):
        """
        Update MET parameters based on time step.
        """
        self.start_met = int(self.num_steps * 0.2) # Starting MET process at 20% of time steps
        self.full_met = int(self.num_steps * 0.6) # Boosting MET process at 60% time steps
        self.noise_list = [1.2, 0.8, 0.4] # List of noise values to apply
        self.interaction_radius_list = [self.bbox_size * 0.1, self.bbox_size * 0.8]
        if t < self.start_met:
            self.sim_met = False # Flag whether MET is active
            self.interaction_radius = self.interaction_radius_list[0] # Small interaction radius
            self.noise = self.noise_list[0]  # Increased randomness
        elif self.start_met <= t < self.full_met:
            self.sim_met = True
            self.interaction_radius = 3 + (t - self.start_met) * 0.1  # Gradually increasing radius
            self.noise = self.noise_list[1] # Less random
        else:
            self.sim_met = True
            self.interaction_radius = self.interaction_radius_list[1]  # Increased interaction radius
            self.noise = self.noise_list[2]  # Less random, more cohesive movement

    def extract_global_features(self, trajectories: np.ndarray) -> dict:
        """Extacts features from simulation and stores in dictionary."""
        # Basic speed and displacement
        displacements = np.diff(trajectories, axis=0)
        speeds = np.linalg.norm(displacements, axis=2)
        avg_speed = speeds.mean()

        # Pairwise distances at final time step
        final_pos = trajectories[-1]
        dists = np.linalg.norm(final_pos[:, None, :] - final_pos[None, :, :], axis=-1)
        avg_pairwise_dist = dists[np.triu_indices_from(dists, k=1)].mean()

        return {
            "avg_speed": avg_speed,
            "avg_pairwise_dist": avg_pairwise_dist,
            # Add more as needed
        }
    
    def save_simulation_data(self, save_dir: str = "./sim_data") -> None:
        """Save simulation data to an npz file with metadata and features."""
        os.makedirs(save_dir, exist_ok=True)

        self.features = self.extract_global_features(self.trajectories)
        
        self.metadata = {"N": self.N, 
                         "num_steps": self.num_steps,
                         "step_size": self.step_size,
                         "seed": self.seed,
                         "sim_met": self.sim_met,
                         "start_met": self.start_met,
                         "full_met": self.full_met,
                         "start_radius": self.interaction_radius_list[0],
                         "end_radius": self.interaction_radius_list[1]}
        
        file_path = os.path.join(save_dir, (self.sim_name + ".npz"))
        
        print(f"Saving simulation data to {file_path}")
        np.savez(file_path, 
                 trajectories=self.trajectories, 
                 features=self.features, 
                 metadata=self.metadata) # Saving simulation data

    def simulate(self) -> None:
        """Simulate MET using agent-based model."""
        # Loop over time steps
        for t in range(self.num_steps): 
            self.update_met_state(t=t) # Updating state of MET process

            # Update direction angles, positions, trajectories/paths
            self.angles = self.get_new_angles()
            self.positions[:, 0] += self.step_size * np.cos(self.angles)
            self.positions[:, 1] += self.step_size * np.sin(self.angles)
            self.apply_bounds() # Boudning cells inside region
            
            self.trajectories[t] = self.positions

            # Loop over cells
            for i in range(self.N):
                # Update cell plots
                self.scatters[i].set_data([self.positions[i, 0]], [self.positions[i, 1]]) 
                
                if self.show_paths:
                    self.trails[i].set_data(self.trajectories[:t+1, i, 0], self.trajectories[:t+1, i, 1])

            met_state = "None" if not self.sim_met else (
                "Partial" if t < self.full_met else "Full")
            self.ax.set_title(f"MET Simulation (t={t}, MET stage = {met_state})")
            plt.pause(0.1) # Pausing 
        
        # Saving simulation figure
        if self.save_sim_fig:
            plt.savefig(os.path.join("./sim_figs", (self.sim_name + ".png")))
        
        plt.clf() # Close figure

        # Saving simulation data
        if self.save_sim_data:
            self.save_simulation_data()


def main():
    # Sim parameters
    N = 10
    num_steps = 100
    step_size = 1
    sim_met = True
    interaction_radius = 2
    
    met_sim = SimMET(N=N, num_steps=num_steps, step_size=step_size, sim_met=sim_met, interaction_radius=interaction_radius, bbox_size=10) # Simulation object
    met_sim.simulate() # Running simulation

if __name__ == "__main__":
    main()