import torch
import torch.nn as nn
import torch.nn.functional as F

class Glioma3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Glioma3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        
        # Calculate the flattened size (32 * 64 * 64 * 64)
        flattened_size = 32 * 64 * 64 * 64
        self.fc1 = nn.Linear(flattened_size, 128)  # Match the size here
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        print(f"Shape after conv and pool: {x.shape}")  # Debug shape
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




