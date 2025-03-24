import os
from datetime import datetime
import nibabel as nib
import numpy as np
import pandas as pd
from biological_model import BiologicalModel
import multiprocessing
from equation_constant import EquationConstant

def read_excel_data(excel_path):
    lptdg_sheet = pd.read_excel(excel_path, sheet_name='UCSF-LPTDG', header=None)  # No header

    column_p_index = 15
    increased_patients = lptdg_sheet[lptdg_sheet[column_p_index].str.contains("Increased", case=False, na=False)]
    column_a_index = 0 
    patient_ids = increased_patients[column_a_index].tolist()

    clinical_info_sheet = pd.read_excel(excel_path, sheet_name='Clinical Info', header = None)
    days_between_scans = []
    for patient_id in patient_ids:
        matching_row = clinical_info_sheet[clinical_info_sheet[0] == patient_id]
        days = round(matching_row.iloc[0, 2], 2) 
        days_between_scans.append(days)

    return patient_ids, days_between_scans

def read_csv_data(volume_csv_path, clinical_csv_path):
    volume_df = pd.read_csv(volume_csv_path, header=None, encoding='latin1')
    column_p_index = 15  # tumor growth
    increased_patients = volume_df[volume_df[column_p_index].astype(str).str.contains("Increased", case=False, na=False)]
    column_a_index = 0  # patient ID
    patient_ids = increased_patients[column_a_index].tolist()

    clinical_df = pd.read_csv(clinical_csv_path, header=None, encoding='latin1')
    days_between_scans = []

    for patient_id in patient_ids:
        matching_row = clinical_df[clinical_df[0] == patient_id]
        if not matching_row.empty:
            days = round(float(matching_row.iloc[0, 2]), 2)  # time between scans
            days_between_scans.append(days)
        else:
            days_between_scans.append(None)

    return patient_ids, days_between_scans

def automate_tumor_growth(file_paths, target_days, output_dir="output"):
    model = BiologicalModel.instance()
    model.without_app = True
    mri_data = model.load_mri_data(file_paths)

    initial_tumor_mask = mri_data['glistrboost'] > 0
    brain_mask = model.create_brain_mask(mri_data['flair'])

    print("Generating diffusion map using ants...")
    process = multiprocessing.Process(
        target=model.run_ants_diffusion_map,
        args=(file_paths["t1"], model.grey_diffusion_rate, model.white_diffusion_rate, model.csf_diffusion_rate)
    )
    process.start()
    process.join()
    diffusion_map = np.load('diffusion_map.npy')
    print("Diffusion map loaded successfully.")

    needed_time_steps = int(target_days / EquationConstant.TIME_STEP) 
    print(f"Simulating {target_days} days of tumor growth using {needed_time_steps} time steps...")

    full_tumor_mask = model.simulate_growth(
        initial_tumor_mask,
        diffusion_rate=diffusion_map,
        reaction_rate=model.reaction_rate,
        time_steps=needed_time_steps,  # Use the calculated number of steps
        brain_mask=brain_mask 
    )
    os.makedirs(output_dir, exist_ok=True)
    first_segmentation_name = os.path.basename(file_paths['glistrboost'])
    first_segmentation_name = os.path.splitext(first_segmentation_name)[0]
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{first_segmentation_name}_at_{target_days}_days_on_{current_date}.nii"
    output_path = os.path.join(output_dir, output_filename)

    model.save_tumor_mask_as_nii(full_tumor_mask, file_paths['flair'], output_path)
    print(f"Tumor growth simulation completed. Mask saved to {output_path}")

if __name__ == "__main__":
    print("Starting")
    excel_path = r"UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.xlsx"
    csv_path = r"UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL.csv"
    csv_clinical_path = r"UCSF_PostopGlioma_Table S1 R1 V5.0_UNBLINDED_FINAL_clinical_info.csv"
    # patient_ids, days_between_scans = read_excel_data(excel_path)
    patient_ids, days_between_scans = read_csv_data(csv_path, csv_clinical_path)
    for i in range(len(patient_ids)):
        print(f"Processing patient {patient_ids[i]} with {days_between_scans[i]} days between scans...")
        file_paths = {
        'flair': rf"UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0\{patient_ids[i]}\{patient_ids[i]}_time1_flair.nii.gz",
        't1': rf"UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0\{patient_ids[i]}\{patient_ids[i]}_time1_t1.nii.gz",
        'glistrboost': rf"UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0\{patient_ids[i]}\{patient_ids[i]}_time1_seg.nii.gz",
        'seg2': rf"UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0\{patient_ids[i]}\{patient_ids[i]}_time2_seg.nii.gz"
        }
        automate_tumor_growth(file_paths, target_days=days_between_scans[i], output_dir="output")
