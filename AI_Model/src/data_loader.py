import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split


def load_nifti_file(filepath):
    """Load a NIfTI file and return the image array."""
    img = nib.load(filepath)
    return img.get_fdata()


def preprocess_image(image, target_shape=(128, 128, 128)):
    """Resize and normalize image to the target shape."""
    scale_factors = [t / s for t, s in zip(target_shape, image.shape)]
    resized_img = zoom(image, scale_factors, order=1)  # Linear interpolation
    normalized_img = (resized_img - np.mean(resized_img)) / np.std(resized_img)
    return normalized_img


def process_and_save_images(input_dir, output_dir, target_shape=(128, 128, 128)):
    """Process all NIfTI files and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            image = load_nifti_file(filepath)
            processed_image = preprocess_image(image, target_shape)
            np.save(os.path.join(output_dir, filename.replace('.nii.gz', '.npy')), processed_image)
    print(f"All files processed and saved to {output_dir}.")


def prepare_data(data_dir, target_shape=(128, 128, 128)):
    """Load, preprocess, and split data into training and validation sets."""
    images = []
    labels = []  # Placeholder for labels, if available
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            images.append(np.load(os.path.join(data_dir, filename)))
            labels.append(0)  # Replace with actual label-loading logic
    return train_test_split(np.array(images), np.array(labels), test_size=0.2, random_state=42)
