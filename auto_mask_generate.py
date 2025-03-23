import os
from datetime import datetime
import nibabel as nib
import numpy as np
from biological_model import BiologicalModel
import multiprocessing
from equation_constant import EquationConstant

def automate_tumor_growth(file_paths, target_days=5, output_dir="output"):
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
    output_filename = f"{first_segmentation_name}_grown_{current_date}.nii"
    output_path = os.path.join(output_dir, output_filename)

    model.save_tumor_mask_as_nii(full_tumor_mask, file_paths['flair'], output_path)
    print(f"Tumor growth simulation completed. Mask saved to {output_path}")

if __name__ == "__main__":
    file_paths = {
        'flair': r"",
        't1': r"",
        'glistrboost': r"",
        'seg2': r""
    }
    automate_tumor_growth(file_paths, target_days=5, output_dir="output")