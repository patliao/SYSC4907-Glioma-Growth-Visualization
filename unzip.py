import os
import gzip
import shutil

# Define the base directory (relative to the script's location)
base_dir = r"D:\cap\SYSC4907-Glioma-Growth-Visualization\data"

# List of patient directories
patients = [

]

# Files to decompress for each patient
files_to_decompress = [
    "{patient}_time1_flair.nii.gz",
    "{patient}_time1_seg.nii.gz",
    "{patient}_time2_flair.nii.gz",
    "{patient}_time2_seg.nii.gz",
]

# Loop through each patient
for patient in patients:
    # Path to the patient's directory
    patient_dir = os.path.join(base_dir, patient)
    
    # Check if the patient directory exists
    if not os.path.exists(patient_dir):
        print(f"Patient directory not found: {patient_dir}")
        continue
    
    # Loop through the files to decompress
    for file_pattern in files_to_decompress:
        # Format the file name with the patient ID
        gz_file_name = file_pattern.format(patient=patient)
        gz_file_path = os.path.join(patient_dir, gz_file_name)
        
        # Check if the .gz file exists
        if not os.path.exists(gz_file_path):
            print(f"File not found: {gz_file_path}")
            continue
        
        # Define the output file path (remove .gz extension)
        nii_file_path = os.path.join(patient_dir, gz_file_name[:-3])  # Remove .gz
        
        # Decompress the .gz file
        try:
            with gzip.open(gz_file_path, 'rb') as gz_file:
                with open(nii_file_path, 'wb') as nii_file:
                    shutil.copyfileobj(gz_file, nii_file)
            print(f"Decompressed: {gz_file_path} -> {nii_file_path}")
        except Exception as e:
            print(f"Failed to decompress {gz_file_path}: {e}")

print("Decompression complete!")

# Loop through each patient
for patient in patients:
    # Path to the patient's directory
    patient_dir = os.path.join(base_dir, patient)
    
    # Check if the patient directory exists
    if not os.path.exists(patient_dir):
        print(f"Patient directory not found: {patient_dir}")
        continue
    
    # Walk through the directory and delete .gz files
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            if file.endswith(".gz"):
                gz_file_path = os.path.join(root, file)
                try:
                    os.remove(gz_file_path)
                    print(f"Deleted: {gz_file_path}")
                except Exception as e:
                    print(f"Failed to delete {gz_file_path}: {e}")

print("Deletion of .gz files complete!")