"""
File for MET dataset class object.
Pablo Curiel
May 2025
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class METDataset(Dataset):
    """Class for loading in simulated data."""
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.label_keys = ['start_met', 'full_met', 'noise_1', 'noise_2', 'noise_3', 'start_radius', 'end_radius']

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Load in a sample (simulation)."""
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True)

        trajectories = data['trajectories'].astype(np.float32)  # Shape: (T, N, 2)
        trajectories = torch.tensor(trajectories).permute(1, 0, 2)  # New shape: (N, T, 2)

        metadata_dict = data['metadata'].item()
        gt = torch.tensor([metadata_dict[k] for k in self.label_keys], dtype=torch.float32) # GT labels

        return trajectories, gt, idx