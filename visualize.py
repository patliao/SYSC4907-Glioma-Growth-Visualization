import os
import nibabel as nib
import numpy as np
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

# Function to visualize the results
def visualize_results(patient_id):
    print(f"Visualizing results for patient: {patient_id}")
    
    # Load the images
    time2_flair_path = os.path.join(base_dir, patient_id, f"{patient_id}_time2_flair.nii")
    time1_seg_path = os.path.join(base_dir, patient_id, f"{patient_id}_time1_seg.nii")
    time2_seg_path = os.path.join(base_dir, patient_id, f"{patient_id}_time2_seg.nii")
    predicted_seg_path = os.path.join(base_dir, patient_id, f"{patient_id}_predicted_seg.nii")

    time2_flair = load_nii_file(time2_flair_path)
    time1_seg = normalize_segmentation(load_nii_file(time1_seg_path))
    time2_seg = normalize_segmentation(load_nii_file(time2_seg_path))
    predicted_seg = normalize_segmentation(load_nii_file(predicted_seg_path))

    time2_flair_normalized = normalize_image(time2_flair)

    # Create a figure and axis for visualization
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))  # 4 columns for FLAIR, T1, T2, and predicted
    plt.subplots_adjust(bottom=0.25)  # Adjust the bottom margin for the slider

    # Display the initial slice
    slice_idx = predicted_seg.shape[2] // 2  # Start at middle slice
    flair_slice = time2_flair_normalized[:, :, slice_idx]
    t1_seg_slice = time1_seg[:, :, slice_idx]
    t2_seg_slice = time2_seg[:, :, slice_idx]
    pred_slice = predicted_seg[:, :, slice_idx]

    # Plot the FLAIR image, Time 1 segmentation, Time 2 segmentation, and predicted mask
    flair_img = ax[0].imshow(flair_slice, cmap="gray")
    ax[0].set_title("FLAIR (Time 2)")

    t1_seg_img = ax[1].imshow(t1_seg_slice, cmap="gray")
    ax[1].set_title("Ground Truth (Time 1)")

    t2_seg_img = ax[2].imshow(t2_seg_slice, cmap="gray")
    ax[2].set_title("Ground Truth (Time 2)")

    pred_img = ax[3].imshow(pred_slice, cmap="gray")
    ax[3].set_title("Predicted Segmentation")

    # Add a slider for slice selection
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Define the slider position
    slice_slider = Slider(
        ax_slider, 
        'Slice', 
        0, 
        predicted_seg.shape[2] - 1, 
        valinit=slice_idx, 
        valstep=1
    )

    # Update function for the slider
    def update(val):
        slice_idx = int(slice_slider.val)
        flair_slice = time2_flair_normalized[:, :, slice_idx]
        t1_seg_slice = time1_seg[:, :, slice_idx]
        t2_seg_slice = time2_seg[:, :, slice_idx]
        pred_slice = predicted_seg[:, :, slice_idx]
        
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

if __name__ == "__main__":
    # List of patients to visualize
    patients = ["100118", "100121", "100016", "100017", "100019"]
    
    for patient in patients:
        try:
            visualize_results(patient)
        except Exception as e:
            print(f"Error visualizing patient {patient}: {str(e)}")