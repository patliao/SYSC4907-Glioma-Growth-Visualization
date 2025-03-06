import nibabel as nib
import numpy as np
from skimage.transform import resize

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

# how well the predicted tumor and actual tumors overlap
# numerator = 2 * number of common tumor voxels (match)
# denom = number of voels marked as tumor summed in both sets
def dice_similarity_coeff(modelled_tumor, actual_tumor):
    common_tumor_cells = np.sum(modelled_tumor * actual_tumor)
    total_predicted_cells = np.sum(modelled_tumor)
    total_actual_cells = np.sum(actual_tumor)
    dsc = (2 * common_tumor_cells) / (total_predicted_cells + total_actual_cells)
    return dsc

# main code
modelled_tumor = load_tumor_mask("")
actual_tumor = load_tumor_mask("")
resized_modelled_tumor = assert_mask_shapes_match(modelled_tumor, actual_tumor)
print(f"Sensitivity: {sensitivity(resized_modelled_tumor, actual_tumor)}")
print(f"DSC: {dice_similarity_coeff(resized_modelled_tumor, actual_tumor)}")



