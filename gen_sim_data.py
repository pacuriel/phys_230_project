"""
File to generate a diverse set of simulation data.
Pablo Curiel
May 2025
"""
import numpy as np
import os
from tqdm import tqdm

from rw_model import SimMET

def generate_sim_data(num_samples: int, 
                      N: int = 40,
                      num_steps: int = 100,
                      step_size: float = 1.0,
                      bbox_size: int = 10.0):
    
    # os.makedirs(save_dir, exist_ok=True)

    # Loop over each number of sample to generate
    for i in tqdm(range(num_samples)):
        seed = np.random.randint(1e6) # Unique sim seed

        # Randomize parameters
        start_met_frac = np.random.uniform(0.1, 0.4)
        full_met_frac = np.random.uniform(0.5, 0.9)
        start_met = int(start_met_frac * num_steps)
        full_met = int(full_met_frac * num_steps)

        noise_list = sorted(np.random.uniform(0.2, 1.5, size=3), reverse=True)  # Descending noise

        # Interaction radius list
        interaction_radius_list = sorted([
            np.random.uniform(bbox_size * 0.01, bbox_size * 0.2),
            np.random.uniform(bbox_size * 0.5, bbox_size * 0.9)
        ])

        # Sim object
        sim = SimMET(
            N=N,
            num_steps=num_steps,
            step_size=step_size,
            bbox_size=bbox_size,
            seed=seed,
            display_plot=False
        )

        # Inject varied parameters
        sim.start_met = start_met
        sim.full_met = full_met
        sim.noise_list = noise_list
        sim.interaction_radius_list = interaction_radius_list

        sim.simulate()


def main():
    num_samples = 9000

    generate_sim_data(num_samples=num_samples)

if __name__ == "__main__":
    main()