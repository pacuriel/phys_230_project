"""
Simple MLP for physics project.
Pablo Curiel
May 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=8000, hidden_dim_1=64, hidden_dim_2=64, output_dim=7):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1) # Flatten: (B, N, T, 2) -> (B, N*T*2)
        x = F.relu(self.fc1(x)) # First hidden layer
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) # Second hidden layer
        x = self.dropout(x)
        x = self.fc3(x) # Output layer (no activation)
        return x

if __name__ == "__main__":
    N = 40
    T = 100
    B = 32
    model = MLP()

    # Dummy input tensor representing one data sample
    sample_input = torch.randn(B, N, T, 2)
    print(sample_input.shape)
    output = model(sample_input)
    print(output.shape)
    # print("Predicted random walk parameters:", output)