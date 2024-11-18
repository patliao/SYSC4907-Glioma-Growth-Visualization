import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

# Step 1: Load MRI Data
def load_mri_data(file_paths):
    return {key: nib.load(file).get_fdata() for key, file in file_paths.items()}

# Step 2: Resize the tumor mask to match the slice shape
def resize_mask_to_slice(tumor_mask, slice_shape):
    resized_mask = resize(tumor_mask, slice_shape, order=0, preserve_range=True, anti_aliasing=False)
    return resized_mask.astype(bool)

# Step 3: Simulate Tumor Growth using Reaction-Diffusion
def simulate_growth(initial_mask, diffusion_rate, reaction_rate, time_steps):
    mask = initial_mask.copy().astype(float)
    for _ in range(time_steps):
        # Apply Gaussian filter for diffusion and add reaction (growth)
        mask = gaussian_filter(mask, sigma=diffusion_rate) + reaction_rate * mask * (1 - mask)
        mask = np.clip(mask, 0, 1)  # Keep values in range
    return mask > 0.5  # Threshold to keep mask as binary

# Step 4: Interactive Visualization with Slice, Time Sliders, and Overlay Toggle
def interactive_growth_visualization(mri_data):
    sagittal_slice_idx = mri_data['flair'].shape[0] // 2  # Start at the middle slice along the z-axis (sagittal)
    coronal_slice_idx = mri_data['flair'].shape[1] // 2  # Start at the middle slice along the x-axis (coronal)
    axial_slice_idx = mri_data['flair'].shape[2] // 2
    # Get the initial tumor mask for sagittal and coronal slices
    initial_tumor_mask_sagittal = mri_data['glistrboost'][sagittal_slice_idx, :, :] > 0
    initial_tumor_mask_coronal = mri_data['glistrboost'][:, coronal_slice_idx, :] > 0
    initial_tumor_mask_axial = mri_data['glistrboost'][:, :, axial_slice_idx] > 0
    # Resize the tumor masks to match the slice dimensions
    tumor_mask_resized_sagittal = resize_mask_to_slice(initial_tumor_mask_sagittal, mri_data['flair'].shape[1:3])
    tumor_mask_resized_coronal = resize_mask_to_slice(initial_tumor_mask_coronal, mri_data['flair'].shape[1:3])
    tumor_mask_resized_axial = resize_mask_to_slice(initial_tumor_mask_axial, mri_data['flair'].shape[:2])
    # Set up the figure and axis
    fig, (ax_sagittal, ax_coronal, ax_axial) = plt.subplots(1, 3, figsize=(14, 7))
    plt.subplots_adjust(left=0.25, bottom=0.35)

    # Initial scan setup for both figures
    current_scan = 'flair'
    scan_slice_sagittal = mri_data[current_scan][sagittal_slice_idx, :, :].T  # sagittal slice (z-axis)
    scan_slice_coronal = mri_data[current_scan][:, coronal_slice_idx, :].T  # coronal slice (x-axis)
    scan_slice_axial = mri_data[current_scan][:, :, axial_slice_idx].T
    
    scan_rgb_sagittal = np.repeat(scan_slice_sagittal[:, :, np.newaxis], 3, axis=2)
    scan_rgb_coronal = np.repeat(scan_slice_coronal[:, :, np.newaxis], 3, axis=2)
    scan_rgb_axial = np.repeat(scan_slice_axial[:, :, np.newaxis], 3, axis=2)

    # Normalize scan_rgb to ensure values are within [0, 1]
    scan_rgb_sagittal = np.clip(scan_rgb_sagittal / np.max(scan_rgb_sagittal), 0, 1)
    scan_rgb_coronal = np.clip(scan_rgb_coronal / np.max(scan_rgb_coronal), 0, 1)
    scan_rgb_axial = np.clip(scan_rgb_axial / np.max(scan_rgb_axial), 0, 1)
    # Display the initial tumor mask (default to FLAIR)
    scan_img_sagittal = ax_sagittal.imshow(scan_rgb_sagittal, origin='lower')
    scan_img_coronal = ax_coronal.imshow(scan_rgb_coronal, origin='lower')
    scan_img_axial = ax_axial.imshow(scan_rgb_axial, origin='lower')
    # Apply initial red overlay for the tumor region
    tumor_overlay_sagittal = tumor_mask_resized_sagittal.T
    tumor_overlay_coronal = tumor_mask_resized_coronal.T
    tumor_overlay_axial= tumor_mask_resized_axial.T
    overlay_on = True  # Control variable for overlay toggle

    # Apply tumor overlays
    scan_rgb_sagittal[tumor_overlay_sagittal, 0] = 1
    scan_rgb_sagittal[tumor_overlay_sagittal, 1] = 0
    scan_rgb_sagittal[tumor_overlay_sagittal, 2] = 0

    scan_rgb_coronal[tumor_overlay_coronal, 0] = 1
    scan_rgb_coronal[tumor_overlay_coronal, 1] = 0
    scan_rgb_coronal[tumor_overlay_coronal, 2] = 0

    scan_rgb_axial[tumor_overlay_axial, 0] = 1
    scan_rgb_axial[tumor_overlay_axial, 1] = 0
    scan_rgb_axial[tumor_overlay_axial, 2] = 0

    scan_img_sagittal.set_data(scan_rgb_sagittal)
    scan_img_coronal.set_data(scan_rgb_coronal)
    scan_img_axial.set_data(scan_rgb_axial)

    # Slider for controlling slice index
    ax_slice_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
    slice_slider = Slider(ax_slice_slider, 'Slice Index', 0, mri_data[current_scan].shape[0] - 1, valinit=sagittal_slice_idx, valstep=1)

    # Slider for controlling time step
    ax_time_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(ax_time_slider, 'Time Step', 0, 50, valinit=0, valstep=1)

    # Checkbox for toggling red overlay
    ax_toggle = plt.axes([0.05, 0.5, 0.15, 0.15])
    toggle_button = CheckButtons(ax_toggle, ['Toggle Overlay'], [overlay_on])

    # RadioButtons for selecting scan type (FLAIR, T1, T1_Gd, etc.)
    ax_radio = plt.axes([0.05, 0.8, 0.15, 0.15])
    radio_button = RadioButtons(ax_radio, ['FLAIR', 'T1', 'T1_Gd', 'T2'])

    # Hide the border around the checkbox
    for label in toggle_button.labels:
        label.set_fontsize(10)
        label.set_color('black')

    ax_toggle.spines['top'].set_visible(False)
    ax_toggle.spines['right'].set_visible(False)
    ax_toggle.spines['left'].set_visible(False)
    ax_toggle.spines['bottom'].set_visible(False)

    # Function to update the overlay toggle
    def toggle_overlay(label):
        nonlocal overlay_on
        overlay_on = not overlay_on
        update(None)  # Re-render the figure with the updated overlay status

    toggle_button.on_clicked(toggle_overlay)

    # Function to update the scan type when radio button is clicked
    def update_scan_type(label):
        nonlocal current_scan
        current_scan = label.lower()  # Update the scan to the selected one
        update(None)  # Re-render the figure with the new scan type

    radio_button.on_clicked(update_scan_type)

    # Update function for the sliders and toggle
    def update(val):
        slice_idx = int(slice_slider.val)
        time_step = int(time_slider.val)

        # Update the selected scan slice for both sagittal and coronal
        scan_slice_sagittal = mri_data[current_scan][slice_idx, :, :].T
        scan_slice_coronal = mri_data[current_scan][:, slice_idx, :].T
        scan_slice_axial = mri_data[current_scan][:, :, slice_idx].T
        scan_rgb_sagittal = np.repeat(scan_slice_sagittal[:, :, np.newaxis], 3, axis=2)
        scan_rgb_coronal = np.repeat(scan_slice_coronal[:, :, np.newaxis], 3, axis=2)
        scan_rgb_axial = np.repeat(scan_slice_axial[:, :, np.newaxis], 3, axis=2)
        # Normalize scan_rgb to ensure values are within [0, 1]
        scan_rgb_sagittal = np.clip(scan_rgb_sagittal / np.max(scan_rgb_sagittal), 0, 1)
        scan_rgb_coronal = np.clip(scan_rgb_coronal / np.max(scan_rgb_coronal), 0, 1)
        scan_rgb_axial = np.clip(scan_rgb_axial / np.max(scan_rgb_axial), 0, 1)
        # Resize the initial mask to match the new slice and simulate growth
        tumor_mask_resized_sagittal = resize_mask_to_slice(mri_data['glistrboost'][slice_idx, :, :] > 0, mri_data[current_scan].shape[1:])
        tumor_mask_resized_coronal = resize_mask_to_slice(mri_data['glistrboost'][:, slice_idx, :] > 0, mri_data[current_scan].shape[1:])
        tumor_mask_resized_axial = resize_mask_to_slice(mri_data['glistrboost'][:, :, slice_idx] > 0, mri_data[current_scan].shape[:2])
        grown_tumor_mask_sagittal = simulate_growth(tumor_mask_resized_sagittal, diffusion_rate=0.5, reaction_rate=0.1, time_steps=time_step)
        grown_tumor_mask_coronal = simulate_growth(tumor_mask_resized_coronal, diffusion_rate=0.5, reaction_rate=0.1, time_steps=time_step)
        grown_tumor_mask_axial = simulate_growth(tumor_mask_resized_axial, diffusion_rate=0.5, reaction_rate=0.1, time_steps=time_step)

        # Apply tumor overlays
        if overlay_on:
            scan_rgb_sagittal[grown_tumor_mask_sagittal.T, 0] = 1
            scan_rgb_sagittal[grown_tumor_mask_sagittal.T, 1] = 0
            scan_rgb_sagittal[grown_tumor_mask_sagittal.T, 2] = 0

            scan_rgb_coronal[grown_tumor_mask_coronal.T, 0] = 1
            scan_rgb_coronal[grown_tumor_mask_coronal.T, 1] = 0
            scan_rgb_coronal[grown_tumor_mask_coronal.T, 2] = 0
            
            scan_rgb_axial[grown_tumor_mask_axial.T, 0] = 1
            scan_rgb_axial[grown_tumor_mask_axial.T, 1] = 0
            scan_rgb_axial[grown_tumor_mask_axial.T, 2] = 0
            
        # Update the images with the new slice and tumor mask
        scan_img_sagittal.set_data(scan_rgb_sagittal)
        scan_img_coronal.set_data(scan_rgb_coronal)
        scan_img_axial.set_data(scan_rgb_axial)

        fig.canvas.draw_idle()

    # Link the update function to the sliders
    slice_slider.on_changed(update)
    time_slider.on_changed(update)
    # Set the background color of the figure and axes
    ax_sagittal.set_facecolor('black')
    ax_coronal.set_facecolor('black')
    ax_axial.set_facecolor('black')

    plt.show()

# Assuming you have a dictionary file_paths of your MRI scan paths
file_paths = {
        'flair': 'C:/Users/firee/Downloads/PKG - BraTS-TCGA-LGG/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-HT-8114/TCGA-HT-8114_1998.10.30_flair.nii',
        'glistrboost': 'C:/Users/firee/Downloads/PKG - BraTS-TCGA-LGG/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-HT-8114/TCGA-HT-8114_1998.10.30_GlistrBoost.nii',
        't1': 'C:/Users/firee/Downloads/PKG - BraTS-TCGA-LGG/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-HT-8114/TCGA-HT-8114_1998.10.30_t1.nii',
        't1_gd': 'C:/Users/firee/Downloads/PKG - BraTS-TCGA-LGG/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-HT-8114/TCGA-HT-8114_1998.10.30_t1Gd.nii',
        't2': 'C:/Users/firee/Downloads/PKG - BraTS-TCGA-LGG/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-HT-8114/TCGA-HT-8114_1998.10.30_t2.nii'
}

# Load the MRI data
mri_data = load_mri_data(file_paths)

# Initialize the interactive visualization
interactive_growth_visualization(mri_data)