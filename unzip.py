import os
import gzip
import shutil

# Define the base directory (relative to the script's location)
base_dir = r"D:\cap\SYSC4907-Glioma-Growth-Visualization\data"

patients = [
    "100006",
    "100008",
    "100011",
    "100016",
    "100017",
    "100019",
    "100020",
    "100022",
    "100026",
    "100027",
    "100038",
    "100040",
    "100041",
    "100044",
    "100049",
    "100050",
    "100051",
    "100057",
    "100061",
    "100063",
    "100065",
    "100066",
    "100067",
    "100072",
    "100081",
    "100083",
    "100084",
    "100087",
    "100088",
    "100093",
    "100095",
    "100097",
    "100102",
    "100104",
    "100108",
    "100109",
    "100110",
    "100114",
    "100116",
    "100118",
    "100121",
    "100125",
    "100131",
    "100132",
    "100140",
    "100142",
    "100147",
    "100148",
    "100152",
    "100153",
    "100155",
    "100156",
    "100158",
    "100161",
    "100165",
    "100167",
    "100169",
    "100171",
    "100176",
    "100181",
    "100184",
    "100186",
    "100187",
    "100188",
    "100189",
    "100190",
    "100191",
    "100193",
    "100196",
    "100199",
    "100202",
    "100206",
    "100209",
    "100210",
    "100213",
    "100219",
    "100220",
    "100221",
    "100224",
    "100227",
    "100229",
    "100230",
    "100232",
    "100236",
    "100238",
    "100239",
    "100241",
    "100245",
    "100246",
    "100248",
    "100249",
    "100251",
    "100253",
    "100256",
    "100259",
    "100260",
    "100267",
    "100273",
    "100276",
    "100277",
    "100281",
    "100282",
    "100283",
    "100285",
    "100286",
    "100289",
    "100290",
    "100291",
    "100292",
    "100295",
    "100298",
    "100302"
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