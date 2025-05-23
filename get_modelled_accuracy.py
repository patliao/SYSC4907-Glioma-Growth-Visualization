import nibabel as nib
import numpy as np
from skimage.transform import resize
from scipy.spatial.distance import directed_hausdorff
import os
import csv

def load_tumor_mask(path_to_nii):
    tumor_mask = nib.load(path_to_nii)
    binary_mask = tumor_mask.get_fdata() > 0
    return binary_mask.astype(bool)

def assert_mask_shapes_match(predicted_mask, actual_mask):
    if predicted_mask.shape != actual_mask.shape:
        resized_predicted_mask = resize(
            predicted_mask, 
            actual_mask.shape, 
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(bool)
        print("Resizing needed.")
        return resized_predicted_mask

    print("Sizes match.")
    return predicted_mask

# how well the model predicts actual tumor cells
def sensitivity(modelled_tumor, actual_tumor):
    TP = np.sum((modelled_tumor == 1) & (actual_tumor == 1))
    FN = np.sum((modelled_tumor == 0) & (actual_tumor == 1))
    return TP / (TP + FN)

# how much the predicted and actual tumors overlap
def jaccard_index(modelled_tumor, actual_tumor):
    TP = np.sum((modelled_tumor == 1) & (actual_tumor == 1))
    FP = np.sum((modelled_tumor == 1) & (actual_tumor == 0))
    FN = np.sum((modelled_tumor == 0) & (actual_tumor == 1))
    return TP / (TP + FP + FN)

# how well the predicted tumor and actual tumors overlap
# numerator = 2 * number of common tumor voxels (match)
# denom = number of voxels marked as tumor summed in both sets
def dice_similarity_coeff(modelled_tumor, actual_tumor):
    common_tumor_cells = np.sum(modelled_tumor * actual_tumor)
    total_predicted_cells = np.sum(modelled_tumor)
    total_actual_cells = np.sum(actual_tumor)
    dsc = (2 * common_tumor_cells) / (total_predicted_cells + total_actual_cells)
    return dsc

# normalizes directed Hausdorff distances by the number of ground truth voxels
# adjusts for tumor size so numbers are fairer
# "how bad is the error on average considering the tumor size?"
def balanced_average_hausdorff(modelled_tumor, actual_tumor):
    modelled_tumor_array = np.argwhere(modelled_tumor)  # predicted segmentation (S)
    actual_tumor_array = np.argwhere(actual_tumor)  # ground truth (G)

    # number of ground truth voxels
    G = len(actual_tumor_array)

    if G == 0:
        raise ValueError("Ground truth mask has no tumor voxels")
    
    G_to_S = directed_hausdorff(actual_tumor_array, modelled_tumor_array)[0]  # ground truth to prediction
    S_to_G = directed_hausdorff(modelled_tumor_array, actual_tumor_array)[0]  # prediction to ground truth

    BAHD = (G_to_S / G + S_to_G / G) / 2

    return BAHD

# main code
# modelled_tumor = load_tumor_mask(r"")
# actual_tumor = load_tumor_mask(r"")
# resized_modelled_tumor = assert_mask_shapes_match(modelled_tumor, actual_tumor)
# print(f"Sensitivity: {sensitivity(resized_modelled_tumor, actual_tumor)}")
# print(f"JI: {jaccard_index(resized_modelled_tumor, actual_tumor)}")
# print(f"DSC: {dice_similarity_coeff(resized_modelled_tumor, actual_tumor)}")
# print(f"Balanced Average Hausdorff Distance (BAHD): {balanced_average_hausdorff(resized_modelled_tumor, actual_tumor)}mm")

output_dir = "output"
patient_data_dir = "UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0"
results = []

for filename in os.listdir(output_dir):
    modelled_growth_file_path = os.path.join(output_dir, filename)
    print(f"Using modelled tumor file: {modelled_growth_file_path}")
    output_filename_info = filename.split("_")
    patient_id = output_filename_info[0]

    if not patient_id.isdigit():
        continue

    patient_folder_path = os.path.join(patient_data_dir, patient_id)
    for f in os.listdir(patient_folder_path):
        if f.endswith("time2_seg.nii.gz"):
            actual_progressed_tumor_file = os.path.join(patient_folder_path, f)
            print(f"Using actual tumor file: {actual_progressed_tumor_file}")
            break
    
    try:
        modelled_tumor = load_tumor_mask(modelled_growth_file_path)
        actual_tumor = load_tumor_mask(actual_progressed_tumor_file)
        modelled_tumor = assert_mask_shapes_match(modelled_tumor, actual_tumor)

        sensitivity_result = sensitivity(modelled_tumor, actual_tumor)
        ji = jaccard_index(modelled_tumor, actual_tumor)
        dsc = dice_similarity_coeff(modelled_tumor, actual_tumor)
        bahd = balanced_average_hausdorff(modelled_tumor, actual_tumor)

        result = {
            "Patient ID": patient_id,
            "Sensitivity": sensitivity_result,
            "Jaccard Index": ji,
            "Dice Similarity Coefficient": dsc,
            "Balanced Avg Hausdorff Dist (mm)": bahd
        }
        results.append(result)

    except Exception as e:
            print(f"Error processing {patient_id}: {e}")

if results:
    output_csv = "equation_results/modelling_results_DgDwdiv5_Dw0.088_R0.029.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll results saved to: {output_csv}")
else:
    print("No results to write.")