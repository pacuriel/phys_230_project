import numpy as np
import matplotlib.pyplot as plt

class SimMET:
    """Class to simulate MET process"""
    def __init__(self,
                 N: int,
                 num_steps: int,
                 step_size: float,
                 bbox_size: int = 10) -> None:
        """
        SimMet class constructor to set up simulation.

        Input
            N: number of cells
            num_steps: number of time steps to simulate
            step_size: step cells should take in each direction
            
        """
        self.N = N
        self.num_steps = num_steps
        self.step_size = step_size
        self.bbox_size = bbox_size

        self.set_sim_vars()

    def simulate(self):
        """Simulate MET using agent-based model."""
        # Loop over time steps
        for t in range(self.num_steps): 
            # Update direction angles, positions, trajectories 
            self.angles = np.random.rand(self.N) * 2 * np.pi # Getting random angles

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

    def set_sim_vars(self):
        """Set variables relevant to simulations."""
        # self.positions = np.zeros((self.N, 2)) # Initializing cell positions
        self.positions = np.random.uniform(-self.bbox_size // 2, self.bbox_size // 2, size=(self.N, 2))
        self.angles = np.random.rand(self.N) * 2 * np.pi
        self.trajectories = np.zeros((self.num_steps, self.N, 2))  # to store trails
        
        self.setup_plotting() # Settting plotting variables
        
    def setup_plotting(self):
        """Setting up plotting variables."""
        # Plotting variables
        self.fig, self.ax = plt.subplots()
        self.bbox_size = 10 # Plot size 
        self.ax.set_xlim(-self.bbox_size, self.bbox_size)
        self.ax.set_ylim(-self.bbox_size, self.bbox_size)

        self.colors = plt.cm.jet(np.linspace(0, 1, self.N))  # Unique color for each cell
        
        self.scatters = [self.ax.plot([], [], 'o', color=self.colors[i])[0] for i in range(self.N)]
        
        self.show_paths = True # Flag whether to show the cell paths
        if self.show_paths:
            self.trails = [self.ax.plot([], [], '-', color=self.colors[i], alpha=0.5)[0] for i in range(self.N)]


def main():
    # Sim parameters
    N = 10
    num_steps = 20
    step_size = 0.5
    
    met_sim = SimMET(N=N, num_steps=num_steps, step_size=step_size)
    met_sim.simulate()

if __name__ == "__main__":
    main()