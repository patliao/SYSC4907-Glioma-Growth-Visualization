import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from matplotlib.widgets import Slider
import torch.nn.functional as F

# Debugging function to print shapes and unique values
def debug_print(data, name):
    print(f"Debug: {name} shape = {data.shape}, unique values = {np.unique(data)}")

# Function to visualize a 2D slice of 3D data
def visualize_slice(data, title, slice_idx=None):
    """
    Visualizes a 2D slice of 3D data using an interactive slider to navigate through slices.
    
    Args:
        data (numpy.ndarray): 3D data to visualize.
        title (str): Title of the plot.
        slice_idx (int, optional): Index of the slice to start with. Defaults to middle slice.
    """
    # If no slice index is provided, set it to the middle slice
    if slice_idx is None:
        slice_idx = data.shape[2] // 2
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Initial slice index (start with the middle slice for example)
    init_slice = data.shape[2] // 2

    # Function to update the displayed slice based on the slider
    def update(val):
        slice_index = int(val)  # Convert slider value to an integer
        ax.clear()  # Clear the current axis
        ax.imshow(data[:, :, slice_index], cmap='gray')
        ax.axis('off')  # Hide axes for better visualization
        ax.set_title(f'Slice {slice_index}')
        fig.canvas.draw_idle()  # Redraw the figure

    # Create a slider for selecting the slice index
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])  # [x, y, width, height]
    slice_slider = Slider(
        ax=axfreq,
        label='Slice',
        valmin=0,  # Minimum slice index
        valmax=data.shape[2] - 1,  # Maximum slice index
        valinit=init_slice,  # Initial slice index
        valstep=1  # Step size of 1 to ensure it only goes through integer slices
    )

    # Link the slider to the update function
    slice_slider.on_changed(update)

    # Show the initial slice
    update(init_slice)

    # Display the plot
    plt.show()


def load_patient_modalities(patient_dir, patient_id, time_point):
    """
    Load FLAIR and SEG modalities for a single patient at a specific time point.
    
    Args:
        patient_dir (str): Path to the patient's directory (e.g., 'data/100116').
        patient_id (str): Patient ID (e.g., '100116').
        time_point (str): Time point to load (e.g., 'time1' or 'time2').
    
    Returns:
        dict: A dictionary containing FLAIR and SEG as numpy arrays.
    """
    modalities = ['flair', 'seg']  # Only load FLAIR and SEG
    patient_data = {}
    
    for modality in modalities:
        file_name = f'{patient_id}_{time_point}_{modality}.nii.gz'  # Construct file name
        file_path = os.path.join(patient_dir, file_name)
        
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            patient_data[modality] = nib.load(file_path).get_fdata()
            debug_print(patient_data[modality], f"{modality} at {time_point}")

        else:
            print(f"Warning: {file_path} not found. Skipping {modality}.")
    
    return patient_data

def load_patient_data(patient_dir, patient_id):
    """
    Load data for both time points (time1 and time2) for a single patient.
    
    Args:
        patient_dir (str): Path to the patient's directory (e.g., 'data/100116').
        patient_id (str): Patient ID (e.g., '100116').
    
    Returns:
        dict: A dictionary containing data for both time points.
    """
    time_points = ['time1', 'time2']
    patient_data = {}
    
    for time_point in time_points:
        print(f"Loading data for {patient_id} at {time_point}...")
        patient_data[time_point] = load_patient_modalities(patient_dir, patient_id, time_point)
    
    return patient_data

def normalize(data):
    """
    Normalize the data to the range [0, 1].
    
    Args:
        data (numpy.ndarray): Input 3D MRI data.
    
    Returns:
        numpy.ndarray: Normalized data.
    """
    if np.max(data) == np.min(data):
        print("Warning: Image is constant. Skipping normalization.")
        return data
    
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    debug_print(normalized_data, "Normalized data")
    
    return normalized_data

def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    """
    Resample an image to a new spacing.
    
    Args:
        image (numpy.ndarray): Input 3D MRI data.
        new_spacing (list): Desired voxel spacing (e.g., [1.0, 1.0, 1.0]).
    
    Returns:
        numpy.ndarray: Resampled image.
    """
    sitk_image = sitk.GetImageFromArray(image)
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    
    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = resampler.Execute(sitk_image)
    
    resampled_data = sitk.GetArrayFromImage(resampled_image)
    debug_print(resampled_data, "Resampled data")
    
    return resampled_data

def prepare_dataset(patient_data):
    """
    Prepare the dataset by pairing initial (time1) and follow-up (time2) scans.
    Uses only FLAIR as input.
    
    Args:
        patient_data (dict): Preprocessed data for a single patient.
    
    Returns:
        list: A list of tuples, where each tuple contains (input, target).
    """
    dataset = []
    
    # Get time1 and time2 data
    time1_data = patient_data['time1']
    time2_data = patient_data['time2']
    
    # Input: FLAIR from time1 (shape: [H, W, D])
    input_data = time1_data['flair']
    
    # Target: Segmentation mask from time2 (shape: [H, W, D])
    target_data = time2_data['seg']
    
    dataset.append((input_data, target_data))
    return dataset

def visualize_tumor_growth(time1_seg, time2_seg, slice_idx=None):
    # Ensure time1_seg and time2_seg are 3D arrays
    if time1_seg.ndim == 3 and time2_seg.ndim == 3:
        if slice_idx is None:
            slice_idx = time1_seg.shape[2] // 2  # Set to the middle slice if no index provided
        time1_seg = time1_seg[:, :, slice_idx]
        time2_seg = time2_seg[:, :, slice_idx]
    
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Display initial images for Time1 and Time2
    im1 = ax1.imshow(time1_seg, cmap='gray')
    ax1.set_title(f"Tumor at Time1 - Slice {slice_idx}")
    ax1.axis('off')
    
    im2 = ax2.imshow(time2_seg, cmap='gray')
    ax2.set_title(f"Tumor at Time2 - Slice {slice_idx}")
    ax2.axis('off')
    
    # Create sliders for selecting slices for each timepoint
    ax_slider1 = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider1 = Slider(ax_slider1, 'Slice Time1', 0, time1_seg.shape[2] - 1, valinit=slice_idx, valstep=1)
    
    ax_slider2 = plt.axes([0.1, 0.06, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider2 = Slider(ax_slider2, 'Slice Time2', 0, time2_seg.shape[2] - 1, valinit=slice_idx, valstep=1)
    
    # Function to update the displayed images based on slider values
    def update(val):
        slice_idx1 = int(slider1.val)
        slice_idx2 = int(slider2.val)
        
        # Update the images for both timepoints
        im1.set_data(time1_seg[:, :, slice_idx1])
        im2.set_data(time2_seg[:, :, slice_idx2])
        
        # Update titles to show the current slice indices
        ax1.set_title(f"Tumor at Time1 - Slice {slice_idx1}")
        ax2.set_title(f"Tumor at Time2 - Slice {slice_idx2}")
        
        fig.canvas.draw_idle()  # Refresh the figure
    
    # Attach update function to slider changes
    slider1.on_changed(update)
    slider2.on_changed(update)
    
    # Show the plot
    plt.show()

def to_tensor_dataset(data):
    """
    Convert a list of (input, target) pairs into PyTorch tensors.
    
    Args:
        data (list): List of (input, target) pairs.
    
    Returns:
        TensorDataset: PyTorch dataset.
    """
    inputs = torch.tensor(np.array([sample[0] for sample in data]), dtype=torch.float32)
    targets = torch.tensor(np.array([sample[1] for sample in data]), dtype=torch.float32)
    
    # Add channel dimension (for 2D UNet)
    inputs = inputs.unsqueeze(1)  # Shape: [batch_size, 1, H, W, D]
    targets = targets.unsqueeze(1)  # Shape: [batch_size, 1, H, W, D]
    
    return TensorDataset(inputs, targets)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        # Contracting path
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Expanding path
        self.upconv4 = self.upconv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        
        # Final convolution
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, 2))
        enc3 = self.enc3(F.max_pool3d(enc2, 2))
        enc4 = self.enc4(F.max_pool3d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, 2))
        
        # Expanding path
        upconv4 = self.upconv4(bottleneck)
        upconv4 = torch.cat([upconv4, enc4], 1)
        
        upconv3 = self.upconv3(upconv4)
        upconv3 = torch.cat([upconv3, enc3], 1)
        
        upconv2 = self.upconv2(upconv3)
        upconv2 = torch.cat([upconv2, enc2], 1)
        
        upconv1 = self.upconv1(upconv2)
        upconv1 = torch.cat([upconv1, enc1], 1)
        
        # Final layer: output a binary mask (or multi-class output)
        output = self.final_conv(upconv1)
        return output
    
def save_segmentation_to_nifti(prediction, output_path, reference_image=None):
    """
    Save the model's prediction as a NIfTI file.
    
    Args:
        prediction (torch.Tensor or numpy.ndarray): The predicted 3D segmentation (binary or multi-class).
        output_path (str): Path to save the output NIfTI file.
        reference_image (Optional, torch.Tensor or numpy.ndarray): Reference image to copy affine and header from.
    """
    # If the prediction is a tensor, convert it to numpy
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    # Create a NIfTI image using nibabel
    if reference_image is not None:
        # If reference image is provided, use its affine and header
        affine = reference_image.affine
    else:
        affine = np.eye(4)  # Identity matrix if no reference image is provided
    
    # Create NIfTI image
    nifti_image = nib.Nifti1Image(prediction.astype(np.float32), affine)

    # Save the NIfTI image
    nib.save(nifti_image, output_path)
    print(f"Prediction saved to: {output_path}")

    
# List of patient IDs (only 3 patients for now)
patient_ids = ['100011', '100116', '100118']

# Apply normalization, resampling, and alignment to the loaded data
all_patient_data = {}
for patient_id in patient_ids:
    patient_dir = os.path.join('data', patient_id)
    print(f"Processing patient {patient_id}...")
    patient_data = load_patient_data(patient_dir, patient_id)

    for time_point, modalities in patient_data.items():
        print(f"Preprocessing {time_point}...")
        
        # Normalize data (except for segmentation mask)
        for modality, data in modalities.items():
            if modality != 'seg':
                print(f"Normalizing {modality}...")
                patient_data[time_point][modality] = normalize(data)
        
        # Resample data
        for modality, data in modalities.items():
            print(f"Resampling {modality}...")
            patient_data[time_point][modality] = resample_image(data)
    
    # Store preprocessed data for this patient
    all_patient_data[patient_id] = patient_data

# Prepare dataset for all patients
all_datasets = []
for patient_id, patient_data in all_patient_data.items():
    print(f"Preparing dataset for {patient_id}...")
    dataset = prepare_dataset(patient_data)
    all_datasets.extend(dataset)

# Visualize tumor growth for the first patient
patient_id = patient_ids[0]
time1_seg = all_patient_data[patient_id]['time1']['seg']
time2_seg = all_patient_data[patient_id]['time2']['seg']

# Use the visualize_slice function to display the slices for both timepoints
visualize_slice(time1_seg, "Tumor at Time1")
visualize_slice(time2_seg, "Tumor at Time2")

# Split dataset into training and testing sets
train_data, test_data = train_test_split(all_datasets, test_size=0.33, random_state=42)  # 2 for training, 1 for testing

# Create TensorDatasets
train_dataset = to_tensor_dataset(train_data)
test_dataset = to_tensor_dataset(test_data)

# Create DataLoaders
batch_size = 2  # Smaller batch size for fewer patients
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = UNet3D(in_channels=1, out_channels=1)  # 3D model with 1 input and 1 output channel (FLAIR)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        
        # Pass the full 3D volume to the model
        outputs = model(inputs)  # Model expects the full 3D volume
        
        # Calculate loss
        loss = criterion(outputs, targets)  # Compare the full 3D output with the target
        
        # Backpropagation
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Evaluate the model on the test set
# Assuming you have a trained model and a batch of input data `inputs`
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient computation during inference
    outputs = model(inputs)  # Forward pass through the 3D model
    
    # If the output is a probability map, apply threshold to get a binary segmentation
    predictions = (outputs > 0.5).float()  # Convert to binary mask (threshold = 0.5)
    
    # Save the predicted segmentation as a NIfTI file
    save_segmentation_to_nifti(predictions[0], "output_segmentation.nii.gz", reference_image=inputs[0])