import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def load_nifti_file(filepath):
    """Load a NIfTI file and return the image array."""
    img = nib.load(filepath)
    return img.get_fdata()

def preprocess_image(image, target_shape=(128, 128, 128)):
    scale_factors = [t / s for t, s in zip(target_shape, image.shape)]
    resized_img = zoom(image, scale_factors, order=1)  # Linear interpolation
    normalized_img = (resized_img - np.mean(resized_img)) / np.std(resized_img)  # Normalize
    return normalized_img


def process_subjects(input_dir, output_dir, target_shape=(128, 128, 128)):
    """Process all subjects and save the preprocessed files."""
    os.makedirs(output_dir, exist_ok=True)
    for subject in os.listdir(input_dir):
        subject_path = os.path.join(input_dir, subject)
        if os.path.isdir(subject_path):
            print(f"Processing {subject}...")
            output_subject_path = os.path.join(output_dir, subject)
            os.makedirs(output_subject_path, exist_ok=True)
            for modality in ["flair", "t1", "t1ce", "t2", "seg"]:
                filepath = os.path.join(subject_path, f"{subject}_{modality}.nii.gz")
                if os.path.exists(filepath):
                    image = load_nifti_file(filepath)
                    preprocessed_image = preprocess_image(image, target_shape)
                    np.save(os.path.join(output_subject_path, f"{modality}.npy"), preprocessed_image)
            print(f"Finished processing {subject}.")
    print("All subjects processed.")

if __name__ == "__main__":
    input_dir = "../data/raw"
    output_dir = "../data/processed"
    process_subjects(input_dir, output_dir)
