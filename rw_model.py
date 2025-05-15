"""
File to simulate MET process using random walk model. 
Pablo Curiel
May 2025
"""
import numpy as np
import matplotlib.pyplot as plt

class SimMET:
    """Class to simulate MET process"""
    def __init__(self,
                 N: int,
                 num_steps: int,
                 step_size: float,
                 sim_met: bool = True,
                 interaction_radius: float = 2,
                 bbox_size: int = 10,
                 show_paths: bool = True) -> None:
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

        # MET parameters
        if self.sim_met:
            self.interaction_radius = interaction_radius # Radius for cells to interact


        self.bbox_size = bbox_size
        self.show_paths = show_paths

        self.set_sim_vars()

    def set_sim_vars(self):
        """Set variables relevant to simulations."""
        # self.positions = np.zeros((self.N, 2)) # Initializing cell positions
        self.positions = np.random.uniform(-self.bbox_size // 2, self.bbox_size // 2, size=(self.N, 2))
        self.angles = np.random.rand(self.N) * 2 * np.pi
        self.trajectories = np.zeros((self.num_steps, self.N, 2))  # to store trails

        self.noise = 0.3 # Noise for increased randomness
        
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

    def get_new_angles(self):
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

    def simulate(self) -> None:
        """Simulate MET using agent-based model."""
        # Loop over time steps
        for t in range(self.num_steps): 
            # Update direction angles, positions, trajectories/paths
            self.angles = self.get_new_angles()
            self.positions[:, 0] += self.step_size * np.cos(self.angles)
            self.positions[:, 1] += self.step_size * np.sin(self.angles)
            self.trajectories[t] = self.positions

            # Loop over cells
            for i in range(self.N):
                # Update cell plots
                self.scatters[i].set_data([self.positions[i, 0]], [self.positions[i, 1]]) 
                
                if self.show_paths:
                    self.trails[i].set_data(self.trajectories[:t+1, i, 0], self.trajectories[:t+1, i, 1])
    
            self.ax.set_title(f"Multi-cell Random Walk (t={t})") # Plot title
            plt.pause(0.1) # Pausing 
        
        plt.show()


def main():
    # Sim parameters
    N = 10
    num_steps = 100
    step_size = 0.5
    
    met_sim = SimMET(N=N, num_steps=num_steps, step_size=step_size) # Simulation object
    met_sim.simulate() # Running simulation

if __name__ == "__main__":
    main()