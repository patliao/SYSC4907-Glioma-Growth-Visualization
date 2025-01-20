import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from ipywidgets import interact

# Step 1: Load MRI FLAIR and Segmentation Data
def load_nifti_file(file_path):
    """
    Load a NIfTI file and return its data as a NumPy array.
    """
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

# Example usage:
# mri_flair_path = 'path_to_flair.nii.gz'
# tumor_mask_path = 'path_to_segmentation.nii.gz'
# mri_scan = load_nifti_file(mri_flair_path)
# tumor_mask = load_nifti_file(tumor_mask_path)

# Step 2: Inspect a Slice
def inspect_slice(data, title, slice_index, cmap='gray'):
    """
    Visualize a single slice from a 3D dataset.
    """
    plt.imshow(data[:, :, slice_index], cmap=cmap)
    plt.title(f'{title} - Slice {slice_index}')
    plt.axis('off')
    plt.show()

# Step 3: Flip Tumor Mask
def flip_mask(mask, axis=1):
    """
    Flip the tumor mask along the specified axis.
    """
    return np.flip(mask, axis=axis)

# Step 4: Overlay and Inspect MRI with Tumor Mask
def overlay_mask(scan, mask, slice_index):
    """
    Overlay the tumor mask on the MRI scan and visualize it.
    """
    plt.figure(figsize=(10, 5))
    
    # MRI Scan
    plt.subplot(1, 2, 1)
    plt.imshow(scan[:, :, slice_index], cmap='gray')
    plt.title('MRI Scan Slice')
    plt.axis('off')

    # Overlay flipped mask
    plt.subplot(1, 2, 2)
    plt.imshow(scan[:, :, slice_index], cmap='gray')
    plt.imshow(mask[:, :, slice_index], cmap='Reds', alpha=0.5)  # Red overlay for the mask
    plt.title('MRI with Tumor Mask')
    plt.axis('off')

    plt.show()

# Step 5: Interactive Slider for Visualization
def view_slice(scan, mask, slice_idx):
    """
    Render a slice using sliders for interactive visualization.
    """
    overlay_mask(scan, mask, slice_idx)

# Interactive function setup
def interactive_visualization(scan, mask):
    """
    Create an interactive visualization for exploring slices.
    """
    interact(view_slice, scan=fixed(scan), mask=fixed(mask), slice_idx=(0, scan.shape[2] - 1))

# Example Workflow
# Assuming paths to the NIfTI files are set
mri_flair_path = r"P:\Imaging-v202211\Imaging\Patient-002\week-000\FLAIR.nii\FLAIR.nii"
tumor_mask_path = r"P:\Imaging-v202211\Imaging\Patient-002\week-000\DeepBraTumIA-segmentation\atlas\segmentation\seg_mask.nii"

# Load data
mri_scan = load_nifti_file(mri_flair_path)
tumor_mask = load_nifti_file(tumor_mask_path)
print("Shape of MRI scan:", mri_scan.shape)

# Flip the mask
tumor_mask_flipped = flip_mask(tumor_mask)

# Inspect slices
inspect_slice(mri_scan, 'MRI Scan', 50)
inspect_slice(tumor_mask, 'Original Tumor Mask', 50)
inspect_slice(tumor_mask_flipped, 'Flipped Tumor Mask', 50)

# Interactive visualization
interactive_visualization(mri_scan, tumor_mask_flipped)
