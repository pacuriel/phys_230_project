"""Simple FFN for physics project."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=64, hidden_dim_2=32):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 3)  # 3 outputs for 3 parameters

    def forward(self, x):
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)          # Output layer (no activation bc not probabilities)
        return x

if __name__ == "__main__":
    # Dummy input feature vector size is of size
    num_features = 20
    model = FFN(num_features)

    # Dummy input tensor representing one data sample
    sample_input = torch.randn(1, num_features)
    print(sample_input.shape)
    output = model(sample_input)
    print(output.shape)
    print("Predicted random walk parameters:", output)