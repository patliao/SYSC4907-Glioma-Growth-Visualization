import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Check if GPU is available and set the device
device = torch.device("cpu")  # Force CPU usage
print(f"Using device: {device}")

# Define the base directory
base_dir = r"D:\cap\SYSC4907-Glioma-Growth-Visualization\data"

# List of patient folders
patients = ["100006", "100008", "100011", "100116", "100118","100121"]

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

# Dice Loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1.0
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1.0 - dice

# Combined Dice and BCE Loss
class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets) + self.dice_loss(inputs, targets)

# Load data for each patient
data = []
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
    
    # Normalize the FLAIR images
    time1_flair_normalized = normalize_image(time1_flair)
    time2_flair_normalized = normalize_image(time2_flair)
    
    # Append to data
    data.append({
        "patient": patient,
        "time1_flair": time1_flair_normalized,
        "time1_seg": time1_seg,
        "time2_flair": time2_flair_normalized,
        "time2_seg": time2_seg
    })

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Split the data into training and validation sets
train_data, val_data = train_test_split([d for d in data if d["patient"] != "100121"], test_size=0.2, random_state=42)

# Prepare testing data for 100118 and 100121
test_patients = ["100118", "100121"]
test_data = []

for test_patient in test_patients:
    time1_flair_path = os.path.join(base_dir, test_patient, f"{test_patient}_time1_flair.nii")
    time2_flair_path = os.path.join(base_dir, test_patient, f"{test_patient}_time2_flair.nii")
    time1_seg_path = os.path.join(base_dir, test_patient, f"{test_patient}_time1_seg.nii")
    time2_seg_path = os.path.join(base_dir, test_patient, f"{test_patient}_time2_seg.nii")

    time1_flair = load_nii_file(time1_flair_path)
    time2_flair = load_nii_file(time2_flair_path)
    time1_seg = normalize_segmentation(load_nii_file(time1_seg_path))
    time2_seg = normalize_segmentation(load_nii_file(time2_seg_path))

    test_data.append({
        "patient": test_patient,
        "time1_flair": normalize_image(time1_flair),
        "time1_seg": time1_seg,
        "time2_flair": normalize_image(time2_flair),
        "time2_seg": time2_seg
    })

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

# Simplified U-Net model for CPU training
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        # Encoder with fewer filters
        self.encoder1 = self.conv_block(in_channels, 16)  # Reduced filters
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)
        
        # Decoder with fewer filters
        self.decoder1 = self.upconv_block(128, 64)
        self.decoder2 = self.upconv_block(64, 32)
        self.decoder3 = self.upconv_block(32, 16)
        
        # Final layer
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)
    
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

# Main execution block
if __name__ == "__main__":
    # Create datasets and dataloaders
    train_dataset = GliomaDataset(train_data)
    val_dataset = GliomaDataset(val_data)
    test_dataset = GliomaDataset([test_data])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)  # Set num_workers=0 for Windows
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    # Initialize model, loss function, and optimizer
    model = UNet3D(in_channels=2, out_channels=1).to(device)
    criterion = DiceBCELoss()  # Use Dice + BCE Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)  # Learning rate scheduler

    # Training loop
    num_epochs = 10  # Increased number of epochs
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "glioma_unet3d_best.pth")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the final model
    torch.save(model.state_dict(), "glioma_unet3d_final.pth")
    print("Training complete! Best model saved as 'glioma_unet3d_best.pth'.")