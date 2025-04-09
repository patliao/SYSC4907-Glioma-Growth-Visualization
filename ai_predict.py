import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the base directory
base_dir = r"D:\cap\SYSC4907-Glioma-Growth-Visualization\data"

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

# Pad the tensor to make its dimensions divisible by 16
def pad_to_divisible(tensor, divisible_by=16):
    depth, height, width = tensor.shape[-3:]
    pad_depth = (divisible_by - depth % divisible_by) % divisible_by
    pad_height = (divisible_by - height % divisible_by) % divisible_by
    pad_width = (divisible_by - width % divisible_by) % divisible_by
    
    # Pad the tensor (symmetric padding)
    padded_tensor = F.pad(tensor, (0, pad_width, 0, pad_height, 0, pad_depth))
    return padded_tensor

# Define the model architecture (same as during training)
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

# Load the saved model
device = torch.device("cpu")  # Use "cpu" if no GPU is available
model = UNet3D(in_channels=2, out_channels=1).to(device)
model.load_state_dict(torch.load("glioma_unet3d_final.pth", map_location=device))
model.eval()  # Set the model to evaluation mode

# Prepare the test data
test_patients = [
    "100026",
    "100290",
    "100291",
    "100292",
    "100295",
    "100298",
    "100302"
]
for test_patient in test_patients:
    print(f"Processing test patient: {test_patient}")
    
    # Load and normalize FLAIR images
    time1_flair_path = os.path.join(base_dir, test_patient, f"{test_patient}_time1_flair.nii")
    time2_flair_path = os.path.join(base_dir, test_patient, f"{test_patient}_time2_flair.nii")
    time1_seg_path = os.path.join(base_dir, test_patient, f"{test_patient}_time1_seg.nii")
    time2_seg_path = os.path.join(base_dir, test_patient, f"{test_patient}_time2_seg.nii")

    time1_flair = load_nii_file(time1_flair_path)
    time2_flair = load_nii_file(time2_flair_path)
    time1_seg = normalize_segmentation(load_nii_file(time1_seg_path))
    time2_seg = normalize_segmentation(load_nii_file(time2_seg_path))

    time1_flair_normalized = normalize_image(time1_flair)
    time2_flair_normalized = normalize_image(time2_flair)

    # Stack time1 and time2 FLAIR images as input
    x = torch.tensor(np.stack([time1_flair_normalized, time2_flair_normalized], axis=0), dtype=torch.float32)
    x = pad_to_divisible(x)  # Pad the input tensor
    x = x.unsqueeze(0)  # Add batch dimension (batch size = 1)

    # Generate predictions
    with torch.no_grad():  # Disable gradient computation
        outputs = model(x)

    # Convert the output to a numpy array
    predicted_segmentation = outputs.squeeze().numpy()  # Remove batch and channel dimensions

    # Apply a threshold to the predicted segmentation
    threshold = 0.5
    predicted_segmentation_binary = (predicted_segmentation > threshold).astype(np.float32)

    # Save the predicted segmentation as a NIfTI file
    output_dir = os.path.join(base_dir, test_patient)
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    # Use the affine matrix from the original NIfTI file to maintain spatial information
    original_nii = nib.load(time2_flair_path)
    affine = original_nii.affine

    # Create a NIfTI image from the predicted segmentation
    predicted_nii = nib.Nifti1Image(predicted_segmentation_binary, affine)

    # Save the NIfTI image
    output_path = os.path.join(output_dir, f"{test_patient}_predicted_seg.nii")
    nib.save(predicted_nii, output_path)
    print(f"Predicted segmentation saved at: {output_path}")


    # Create a figure and axis for visualization
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))  # 4 columns for FLAIR, T1, T2, and predicted
    plt.subplots_adjust(bottom=0.25)  # Adjust the bottom margin for the slider

    # Display the initial slice
    slice_idx = 75
    flair_slice = time2_flair[:, :, slice_idx]
    t1_seg_slice = time1_seg[:, :, slice_idx]
    t2_seg_slice = time2_seg[:, :, slice_idx]
    pred_slice = predicted_segmentation_binary[:, :, slice_idx]

    # Plot the FLAIR image, Time 1 segmentation, Time 2 segmentation, and predicted mask
    flair_img = ax[0].imshow(flair_slice, cmap="gray")
    ax[0].set_title("FLAIR (Time 2)")

    t1_seg_img = ax[1].imshow(t1_seg_slice, cmap="gray")
    ax[1].set_title("Ground Truth (Time 1)")

    t2_seg_img = ax[2].imshow(t2_seg_slice, cmap="gray")
    ax[2].set_title("Ground Truth (Time 2)")

    pred_img = ax[1].imshow(pred_slice, cmap="gray")
    ax[1].set_title("Predicted Segmentation")

    # Add a slider for slice selection
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Define the slider position
    slice_slider = Slider(ax_slider, 'Slice', 0, predicted_segmentation_binary.shape[2] - 1, valinit=slice_idx, valstep=1)

    # Update function for the slider
    def update(val):
        slice_idx = int(slice_slider.val)
        flair_slice = time2_flair[:, :, slice_idx]
        t1_seg_slice = time1_seg[:, :, slice_idx]
        t2_seg_slice = time2_seg[:, :, slice_idx]
        pred_slice = predicted_segmentation_binary[:, :, slice_idx]
        
        # Update the images
        flair_img.set_data(flair_slice)
        t1_seg_img.set_data(t1_seg_slice)
        t2_seg_img.set_data(t2_seg_slice)
        pred_img.set_data(pred_slice)
        
        # Redraw the figure
        fig.canvas.draw_idle()

    # Link the slider to the update function
    slice_slider.on_changed(update)

    # Show the plot
    plt.show()