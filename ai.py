import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Check if GPU is available and set the device
device = torch.device("cpu")
print(f"Using device: {device}")

# Define the base directory
base_dir = r"D:\cap\SYSC4907-Glioma-Growth-Visualization\data"

# List of patient folders
patients = ["100006", "100008", "100011", "100116", "100118"]

# Function to load a .nii file
def load_nii_file(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()  # Get the image data as a numpy array
    return data

# Normalize FLAIR images to [0, 1]
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Normalize segmentation masks to [0, 1]
def normalize_segmentation(segmentation):
    return (segmentation > 0).astype(np.float32)  # Convert to binary mask

# Load data for each patient
for patient in patients:
    print(f"Loading data for patient: {patient}")
    
    # Construct file paths (using .nii instead of .nii.gz)
    time1_flair_path = os.path.join(base_dir, patient, f"{patient}_time1_flair.nii")
    time1_seg_path = os.path.join(base_dir, patient, f"{patient}_time1_seg.nii")
    time2_flair_path = os.path.join(base_dir, patient, f"{patient}_time2_flair.nii")
    time2_seg_path = os.path.join(base_dir, patient, f"{patient}_time2_seg.nii")
    
    # Load the files
    time1_flair = load_nii_file(time1_flair_path)
    time1_seg = normalize_segmentation(load_nii_file(time1_seg_path))
    time2_flair = load_nii_file(time2_flair_path)
    time2_seg = normalize_segmentation(load_nii_file(time2_seg_path))
    
    # Print shapes to verify
    print(f"Time1 FLAIR shape: {time1_flair.shape}")
    print(f"Time1 Segmentation shape: {time1_seg.shape}")
    print(f"Time2 FLAIR shape: {time2_flair.shape}")
    print(f"Time2 Segmentation shape: {time2_seg.shape}")
    print("-" * 40)

# Normalize the FLAIR images for each patient
for patient in patients:
    print(f"Normalizing data for patient: {patient}")
    
    # Load the FLAIR images
    time1_flair_path = os.path.join(base_dir, patient, f"{patient}_time1_flair.nii")
    time2_flair_path = os.path.join(base_dir, patient, f"{patient}_time2_flair.nii")
    
    time1_flair = load_nii_file(time1_flair_path)
    time2_flair = load_nii_file(time2_flair_path)
    
    # Normalize the images
    time1_flair_normalized = normalize_image(time1_flair)
    time2_flair_normalized = normalize_image(time2_flair)
    
    # Print min and max values to verify normalization
    print(f"Time1 FLAIR normalized - min: {np.min(time1_flair_normalized)}, max: {np.max(time1_flair_normalized)}")
    print(f"Time2 FLAIR normalized - min: {np.min(time2_flair_normalized)}, max: {np.max(time2_flair_normalized)}")
    print("-" * 40)

# Split the data into training and testing sets
train_patients = ["100006", "100008", "100011", "100116"]  # Added 100006 and 100008
test_patient = "100118"

# Prepare training data
train_data = []
for patient in train_patients:
    print(f"Preparing training data for patient: {patient}")
    
    # Load and normalize FLAIR images
    time1_flair_path = os.path.join(base_dir, patient, f"{patient}_time1_flair.nii")
    time2_flair_path = os.path.join(base_dir, patient, f"{patient}_time2_flair.nii")
    
    time1_flair = load_nii_file(time1_flair_path)
    time2_flair = load_nii_file(time2_flair_path)
    
    time1_flair_normalized = normalize_image(time1_flair)
    time2_flair_normalized = normalize_image(time2_flair)
    
    # Load segmentation masks
    time1_seg_path = os.path.join(base_dir, patient, f"{patient}_time1_seg.nii")
    time2_seg_path = os.path.join(base_dir, patient, f"{patient}_time2_seg.nii")
    
    time1_seg = normalize_segmentation(load_nii_file(time1_seg_path))
    time2_seg = normalize_segmentation(load_nii_file(time2_seg_path))
    
    # Append to training data
    train_data.append({
        "patient": patient,
        "time1_flair": time1_flair_normalized,
        "time1_seg": time1_seg,
        "time2_flair": time2_flair_normalized,
        "time2_seg": time2_seg
    })

# Prepare testing data
print(f"Preparing testing data for patient: {test_patient}")
time1_flair_path = os.path.join(base_dir, test_patient, f"{test_patient}_time1_flair.nii")
time2_flair_path = os.path.join(base_dir, test_patient, f"{test_patient}_time2_flair.nii")

time1_flair = load_nii_file(time1_flair_path)
time2_flair = load_nii_file(time2_flair_path)

time1_flair_normalized = normalize_image(time1_flair)
time2_flair_normalized = normalize_image(time2_flair)

time1_seg_path = os.path.join(base_dir, test_patient, f"{test_patient}_time1_seg.nii")
time2_seg_path = os.path.join(base_dir, test_patient, f"{test_patient}_time2_seg.nii")

time1_seg = normalize_segmentation(load_nii_file(time1_seg_path))
time2_seg = normalize_segmentation(load_nii_file(time2_seg_path))

test_data = {
    "patient": test_patient,
    "time1_flair": time1_flair_normalized,
    "time1_seg": time1_seg,
    "time2_flair": time2_flair_normalized,
    "time2_seg": time2_seg
}

print("Data splitting complete!")

def pad_to_divisible(tensor, divisible_by=16):
    """Pad the tensor to make its dimensions divisible by `divisible_by`."""
    depth, height, width = tensor.shape[-3:]
    pad_depth = (divisible_by - depth % divisible_by) % divisible_by
    pad_height = (divisible_by - height % divisible_by) % divisible_by
    pad_width = (divisible_by - width % divisible_by) % divisible_by
    
    # Pad the tensor (symmetric padding)
    padded_tensor = F.pad(tensor, (0, pad_width, 0, pad_height, 0, pad_depth))
    return padded_tensor

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        # Encoder with fewer filters
        self.encoder1 = self.conv_block(in_channels, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)
        
        # Decoder with fewer filters
        self.decoder1 = self.upconv_block(256, 128)
        self.decoder2 = self.upconv_block(128, 64)
        self.decoder3 = self.upconv_block(64, 32)
        
        # Final layer
        self.final = nn.Conv3d(32, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            self.conv_block(out_channels, out_channels)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool3d(e1, kernel_size=2, stride=2))
        e3 = self.encoder3(F.max_pool3d(e2, kernel_size=2, stride=2))
        e4 = self.encoder4(F.max_pool3d(e3, kernel_size=2, stride=2))
        
        # Decoder
        d1 = self.decoder1(e4)
        d2 = self.decoder2(d1 + e3)
        d3 = self.decoder3(d2 + e2)
        
        # Final layer
        out = self.final(d3 + e1)
        return torch.sigmoid(out)  # Sigmoid for binary segmentation
    
class GliomaDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Stack time1 and time2 FLAIR images as input
        x = torch.tensor(np.stack([sample["time1_flair"], sample["time2_flair"]], axis=0), dtype=torch.float32)
        x = pad_to_divisible(x)  # Pad the input tensor
        
        # Use time2 segmentation mask as target
        y = torch.tensor(sample["time2_seg"], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        y = pad_to_divisible(y)  # Pad the target tensor
        
        return x, y

# Create datasets and dataloaders
train_dataset = GliomaDataset(train_data)
test_dataset = GliomaDataset([test_data])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize model, loss function, and optimizer
model = UNet3D(in_channels=2, out_channels=1).to(device)  # Move model to GPU
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3 # Reduced for quick results
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)  # Move data to GPU
        optimizer.zero_grad()
        
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model for presentation
torch.save(model.state_dict(), "glioma_unet3d.pth")
print("Model saved for presentation!")