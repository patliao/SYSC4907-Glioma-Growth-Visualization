import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the base directory
base_dir = r"D:\cap\SYSC4907-Glioma-Growth-Visualization\data"

# Define the test patient
test_patient = "100118"

# Function to load a .nii file
def load_nii_file(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()  # Get the image data as a numpy array
    return data, img.affine, img.header

# Normalize FLAIR images to [0, 1]
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# Normalize segmentation masks to [0, 1]
def normalize_segmentation(segmentation):
    return (segmentation > 0).astype(np.float32)  # Convert to binary mask

# Pad the tensor to make dimensions divisible by 16
def pad_to_divisible(tensor, divisible_by=16):
    """Pad the tensor to make its dimensions divisible by `divisible_by`."""
    depth, height, width = tensor.shape[-3:]
    pad_depth = (divisible_by - depth % divisible_by) % divisible_by
    pad_height = (divisible_by - height % divisible_by) % divisible_by
    pad_width = (divisible_by - width % divisible_by) % divisible_by
    
    # Pad the tensor (symmetric padding)
    padded_tensor = F.pad(tensor, (0, pad_width, 0, pad_height, 0, pad_depth))
    return padded_tensor

# Load and preprocess the test data
print(f"Loading and preprocessing data for patient: {test_patient}")

# Construct file paths
time1_flair_path = os.path.join(base_dir, test_patient, f"{test_patient}_time1_flair.nii")
time2_flair_path = os.path.join(base_dir, test_patient, f"{test_patient}_time2_flair.nii")

# Load the FLAIR images
time1_flair, affine, header = load_nii_file(time1_flair_path)
time2_flair, _, _ = load_nii_file(time2_flair_path)

# Normalize the FLAIR images
time1_flair_normalized = normalize_image(time1_flair)
time2_flair_normalized = normalize_image(time2_flair)

# Stack time1 and time2 FLAIR images as input
input_data = np.stack([time1_flair_normalized, time2_flair_normalized], axis=0)  # Shape: (2, depth, height, width)

# Convert to PyTorch tensor and add batch dimension
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 2, depth, height, width)
input_tensor = input_tensor.to(device)

# Pad the input tensor to make dimensions divisible by 16
input_tensor = pad_to_divisible(input_tensor)

# Define the 3D U-Net model
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

# Load the trained model
model = UNet3D(in_channels=2, out_channels=1).to(device)
model.load_state_dict(torch.load("glioma_unet3d.pth", map_location=device))
model.eval()  # Set the model to evaluation mode

# Generate predictions
with torch.no_grad():
    output = model(input_tensor)  # Shape: (1, 1, depth, height, width)

# Remove padding from the output
output = output[:, :, :time1_flair.shape[0], :time1_flair.shape[1], :time1_flair.shape[2]]

# Threshold the output to create a binary mask
output_binary = (output > 0.5).float().cpu().numpy()

# Save the prediction as a .nii file
output_nii = nib.Nifti1Image(output_binary.squeeze(), affine, header)
output_path = os.path.join(base_dir, test_patient, f"{test_patient}_prediction.nii")
nib.save(output_nii, output_path)

print(f"Prediction saved to: {output_path}")