import argparse
import multiprocessing
import os, sys
import subprocess
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import platform
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
if platform.system() == "Darwin":
    matplotlib.use("Qt5Agg")
import nibabel as nib
import numpy as np
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from equation_constant import EquationConstant
from scipy.ndimage import distance_transform_edt, center_of_mass
import ants

matplotlib.use('TkAgg')

class BiologicalModel:
    _instance = None

    def __init__(self):
        super().__init__()
        self.mri_data = None
        self.tumor_overlay_axial = None
        self.tumor_overlay_coronal = None
        self.tumor_overlay_sagittal = None
        self.scan_img_axial = None
        self.scan_img_coronal = None
        self.scan_img_sagittal = None
        self.scan_rgb_axial = None
        self.scan_rgb_coronal = None
        self.scan_rgb_sagittal = None
        self.scan_slice_axial = None
        self.scan_slice_coronal = None
        self.scan_slice_sagittal = None
        self.ax_axial = None
        self.ax_coronal = None
        self.ax_sagittal = None
        self.fig = None
        self.tumor_mask_resized_coronal = None
        self.tumor_mask_resized_axial = None
        self.tumor_mask_resized_sagittal = None
        self.initial_tumor_mask_axial = None
        self.initial_tumor_mask_coronal = None
        self.initial_tumor_mask_sagittal = None
        self.axial_slice_idx = None
        self.coronal_slice_idx = None
        self.sagittal_slice_idx = None
        self.file_paths = {}
        self.diffusion_rate = EquationConstant.DIFFUSION_RATE
        self.reaction_rate = EquationConstant.REACTION_RATE
        self.without_app = False
        self.csf_diffusion_rate = EquationConstant.CSF_DIFFUSION_RATE
        self.grey_diffusion_rate = EquationConstant.GREY_DIFFUSION_RATE
        self.white_diffusion_rate = EquationConstant.WHITE_DIFFUSION_RATE

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = BiologicalModel()
        return cls._instance

    def get_diffusion_rate(self):
        return self.diffusion_rate

    def set_csf_diffusion_rate(self, csf_dr):
        self.csf_diffusion_rate = csf_dr

    def get_reaction_rate(self):
        return self.reaction_rate

    def set_reaction_rate(self, reaction_rate):
        self.reaction_rate = reaction_rate


    def get_file_paths(self):
        file_paths = {}

        print("Please select the MRI files for the following sequences:")
        root = Tk()
        root.withdraw()  # Hide the main Tkinter window

        for key in EquationConstant.FILE_KEYS:
            print(f"Select the {key.upper()} file:")
            file_path = self.get_selected_file(key)
            file_paths[key] = file_path

        self.file_paths = file_paths
        root.destroy()
        return file_paths

    def get_selected_file(self, key):
        file_path = askopenfilename(title=f"Select the {key.upper()} file")
        if not file_path:
            print(f"File selection for {key.upper()} was canceled. Exiting.")
            exit()
        return file_path

    def auto_load_files(self):
        file_paths = {}
        current_directory = os.getcwd()
        for root, dirs, files in os.walk(current_directory):
            for file_name in files:
                for file_type in EquationConstant.FILE_KEYS:
                    if file_type in file_name.lower():
                        file_paths[file_type] = os.path.join(root, file_name)

        for key in EquationConstant.FILE_KEYS:
            if key not in file_paths:
                print(f"WARNING: {key} FILE NOT FOUND. PLEASE SELECT FROM FILES.")
                file_path = self.get_selected_file()
                file_paths[key] = file_path

        self.file_paths = file_paths
        return file_paths

    def load_mri_data(self, file_paths, target_shape=(240, 240, 160)):
        """Step 1: Load MRI Data, resize to target shape, and print dimensions."""
        mri_data = {}
        for key, file in file_paths.items():
            # Load the NIfTI file
            nii_img = nib.load(file)
            # Get the data array
            data = nii_img.get_fdata()
            
            # Resize the volume to the target shape
            if data.shape != target_shape:
                print(f"Resizing {key.upper()} scan from {data.shape} to {target_shape}...")
                # Use nearest-neighbor interpolation for binary masks (e.g., GLISTRBOOST)
                order = 0 if key == 'glistrboost' else 1
                data = resize(data, target_shape, order=order, preserve_range=True, anti_aliasing=False)
            
            # Store the resized data in the dictionary
            mri_data[key] = data
            # Print the dimensions of the scan
            print(f"Loaded {key.upper()} scan with dimensions: {data.shape}")
        
        return mri_data

    # Step 2: Resize the tumor mask to match the slice shape
    def resize_mask_to_slice(self, tumor_mask, slice_shape, dtype=bool):
        resized_mask = resize(tumor_mask, slice_shape, order=0, preserve_range=True, anti_aliasing=False)
        return resized_mask.astype(dtype)

    # Step 3: Simulate Tumor Growth using Reaction-Diffusion
    def simulate_growth(self, initial_mask, diffusion_rate, reaction_rate, time_steps, brain_mask):
        mask = initial_mask.copy().astype(float)
        brain_mask_resized = self.resize_mask_to_slice(brain_mask, mask.shape)
        diffusion_map_resized = self.resize_mask_to_slice(diffusion_rate, mask.shape, dtype=float)
        decay = self.get_decay_factor(mask)

        for _ in range(time_steps):
            # Apply Gaussian filter for diffusion and add reaction (growth)
            diffused_mask = gaussian_filter(mask, sigma=1.0) * diffusion_map_resized * decay # adjust each voxel according to the diffusion map
            growth = self.reaction_rate * mask * (1 - mask) * decay

            mask = brain_mask_resized * (mask + diffused_mask + growth) # ensures all contributions are restricted to brain region
            mask = np.clip(mask, 0, 1)  # Keep values in range

        return mask > 0.5  # Threshold to keep mask as binary

    def get_decay_factor(self, mask):
        distance_map = distance_transform_edt(mask == 0)
        decay_factor = np.exp(-distance_map / EquationConstant.LAMBDA)
        return decay_factor

    def time_in_days(self, step):
        max_diffusion = max(
            EquationConstant.CSF_DIFFUSION_RATE,
            EquationConstant.GREY_DIFFUSION_RATE,
            EquationConstant.WHITE_DIFFUSION_RATE
        )
        # mean_diffusion = np.mean([
        #     EquationConstant.CSF_DIFFUSION_RATE,
        #     EquationConstant.GREY_DIFFUSION_RATE,
        #     EquationConstant.WHITE_DIFFUSION_RATE
        # ])
        time_step = (EquationConstant.SPATIAL_RESOLUTION ** 2) / (2 * 3 * max_diffusion)
        return step * time_step

    # Step 4: Interactive Visualization with Slice, Time Sliders, and Overlay Toggle
    def interactive_growth_visualization_2(self, mri_data, cur_scan):
        self.sagittal_slice_idx = mri_data['flair'].shape[0] // 2  # Start at the middle slice along the z-axis (sagittal)
        self.coronal_slice_idx = mri_data['flair'].shape[1] // 2  # Start at the middle slice along the x-axis (coronal)
        self.axial_slice_idx = mri_data['flair'].shape[2] // 2
        # Get the initial tumor mask for sagittal and coronal slices
        self.initial_tumor_mask_sagittal = mri_data['glistrboost'][self.sagittal_slice_idx, :, :] > 0
        self.initial_tumor_mask_coronal = mri_data['glistrboost'][:, self.coronal_slice_idx, :] > 0
        self.initial_tumor_mask_axial = mri_data['glistrboost'][:, :, self.axial_slice_idx] > 0
        # Resize the tumor masks to match the slice dimensions
        self.tumor_mask_resized_sagittal = self.resize_mask_to_slice(self.initial_tumor_mask_sagittal,
                                                                mri_data['flair'].shape[1:3])
        self.tumor_mask_resized_coronal = self.resize_mask_to_slice(self.initial_tumor_mask_coronal, mri_data['flair'].shape[1:3])
        self.tumor_mask_resized_axial = self.resize_mask_to_slice(self.initial_tumor_mask_axial, mri_data['flair'].shape[:2])
        self.second_segmentation_mask_sagittal = mri_data['seg2'][self.sagittal_slice_idx, :, :] > 0
        self.second_segmentation_mask_coronal = mri_data['seg2'][:, self.coronal_slice_idx, :] > 0
        self.second_segmentation_mask_axial = mri_data['seg2'][:, :, self.axial_slice_idx] > 0
        # Resize the second segmentation masks
        self.second_segmentation_mask_resized_sagittal = self.resize_mask_to_slice(
            self.second_segmentation_mask_sagittal,
            mri_data['flair'].shape[1:3]
        )
        self.second_segmentation_mask_resized_coronal = self.resize_mask_to_slice(
            self.second_segmentation_mask_coronal,
            mri_data['flair'].shape[1:3]
        )
        self.second_segmentation_mask_resized_axial = self.resize_mask_to_slice(
            self.second_segmentation_mask_axial,
            mri_data['flair'].shape[:2]
        )
        # Set up the figure and axis
        self.second_segmentation_mask_sagittal = mri_data['seg2'][self.sagittal_slice_idx, :, :] > 0
        self.second_segmentation_mask_coronal = mri_data['seg2'][:, self.coronal_slice_idx, :] > 0
        self.second_segmentation_mask_axial = mri_data['seg2'][:, :, self.axial_slice_idx] > 0
        fig, (self.ax_sagittal, self.ax_coronal, self.ax_axial) = plt.subplots(1, 3, figsize=(50, 50))
        plt.subplots_adjust(left=0.25, bottom=0.35)

        # Initial scan setup for both figures
        # self.current_scan = 'flair'
        self.scan_slice_sagittal = mri_data[cur_scan][self.sagittal_slice_idx, :, :].T  # sagittal slice (z-axis)
        self.scan_slice_coronal = mri_data[cur_scan][:, self.coronal_slice_idx, :].T  # coronal slice (x-axis)
        self.scan_slice_axial = mri_data[cur_scan][:, :, self.axial_slice_idx].T

        self.scan_rgb_sagittal = np.repeat(self.scan_slice_sagittal[:, :, np.newaxis], 3, axis=2)
        self.scan_rgb_coronal = np.repeat(self.scan_slice_coronal[:, :, np.newaxis], 3, axis=2)
        self.scan_rgb_axial = np.repeat(self.scan_slice_axial[:, :, np.newaxis], 3, axis=2)

        # Normalize scan_rgb to ensure values are within [0, 1]
        self.scan_rgb_sagittal = np.clip(self.scan_rgb_sagittal / np.max(self.scan_rgb_sagittal), 0, 1)
        self.scan_rgb_coronal = np.clip(self.scan_rgb_coronal / np.max(self.scan_rgb_coronal), 0, 1)
        self.scan_rgb_axial = np.clip(self.scan_rgb_axial / np.max(self.scan_rgb_axial), 0, 1)

        # Display the initial tumor mask (default to FLAIR)
        self.scan_img_sagittal = self.ax_sagittal.imshow(self.scan_rgb_sagittal, origin='lower')
        self.scan_img_coronal = self.ax_coronal.imshow(self.scan_rgb_coronal, origin='lower')
        self.scan_img_axial = self.ax_axial.imshow(self.scan_rgb_axial, origin='lower')
        # Apply initial red overlay for the tumor region
        self.tumor_overlay_sagittal = self.tumor_mask_resized_sagittal.T
        self.tumor_overlay_coronal = self.tumor_mask_resized_coronal.T
        self.tumor_overlay_axial = self.tumor_mask_resized_axial.T
        self.overlay_on = True  # Control variable for overlay toggle

        self.second_seg_overlay_sagittal = self.second_segmentation_mask_resized_sagittal.T
        self.second_seg_overlay_coronal = self.second_segmentation_mask_resized_coronal.T
        self.second_seg_overlay_axial = self.second_segmentation_mask_resized_axial.T

        # Apply tumor overlays
        self.scan_rgb_sagittal[self.tumor_overlay_sagittal, 0] = 1
        self.scan_rgb_sagittal[self.tumor_overlay_sagittal, 1] = 0
        self.scan_rgb_sagittal[self.tumor_overlay_sagittal, 2] = 0

        # Apply second segmentation overlays in green
        self.scan_rgb_sagittal[self.second_seg_overlay_sagittal, 0] = 0
        self.scan_rgb_sagittal[self.second_seg_overlay_sagittal, 1] = 1
        self.scan_rgb_sagittal[self.second_seg_overlay_sagittal, 2] = 0

        self.scan_rgb_coronal[self.tumor_overlay_coronal, 0] = 1
        self.scan_rgb_coronal[self.tumor_overlay_coronal, 1] = 0
        self.scan_rgb_coronal[self.tumor_overlay_coronal, 2] = 0

        self.scan_rgb_axial[self.tumor_overlay_axial, 0] = 1
        self.scan_rgb_axial[self.tumor_overlay_axial, 1] = 0
        self.scan_rgb_axial[self.tumor_overlay_axial, 2] = 0

        self.scan_img_sagittal.set_data(self.scan_rgb_sagittal)
        self.scan_img_coronal.set_data(self.scan_rgb_coronal)
        self.scan_img_axial.set_data(self.scan_rgb_axial)

        # # Link the update function to the sliders
        # self.slice_slider.on_changed(update)
        # self.time_slider.on_changed(update)
        # Set the background color of the figure and axes
        self.ax_sagittal.set_facecolor('black')
        self.ax_coronal.set_facecolor('black')
        self.ax_axial.set_facecolor('black')
        self.fig = fig
        return fig
        # return [self.ax_sagittal, self.ax_coronal, self.ax_axial]

    def create_brain_mask(self, mri_image):
        brain_mask = mri_image > 0
        return brain_mask

    # Update function for the sliders and toggle
    def update(self, slice_idx, time_step, overlay, cur_scan):

        self.brain_mask_sagittal = self.create_brain_mask(self.mri_data['flair'][slice_idx, :, :])
        self.brain_mask_coronal = self.create_brain_mask(self.mri_data['flair'][:, slice_idx, :])
        self.brain_mask_axial = self.create_brain_mask(self.mri_data['flair'][:, :, slice_idx])

        # Resize the brain mask to the shape of the current slice
        self.brain_mask_resized_sagittal = self.resize_mask_to_slice(self.brain_mask_sagittal,
                                                                         self.mri_data[cur_scan].shape[1:])
        self.brain_mask_resized_coronal = self.resize_mask_to_slice(self.brain_mask_coronal,
                                                                        self.mri_data[cur_scan].shape[1:])
        self.brain_mask_resized_axial = self.resize_mask_to_slice(self.brain_mask_axial,
                                                                      self.mri_data[cur_scan].shape[:2])

        # Update the selected scan slice for both sagittal and coronal
        self.scan_slice_sagittal = self.mri_data[cur_scan][slice_idx, :, :].T
        self.scan_slice_coronal = self.mri_data[cur_scan][:, slice_idx, :].T
        self.scan_slice_axial = self.mri_data[cur_scan][:, :, slice_idx].T
        self.scan_rgb_sagittal = np.repeat(self.scan_slice_sagittal[:, :, np.newaxis], 3, axis=2)
        self.scan_rgb_coronal = np.repeat(self.scan_slice_coronal[:, :, np.newaxis], 3, axis=2)
        self.scan_rgb_axial = np.repeat(self.scan_slice_axial[:, :, np.newaxis], 3, axis=2)
        # Normalize scan_rgb to ensure values are within [0, 1]
        self.scan_rgb_sagittal = np.clip(self.scan_rgb_sagittal / np.max(self.scan_rgb_sagittal), 0, 1)
        self.scan_rgb_coronal = np.clip(self.scan_rgb_coronal / np.max(self.scan_rgb_coronal), 0, 1)
        self.scan_rgb_axial = np.clip(self.scan_rgb_axial / np.max(self.scan_rgb_axial), 0, 1)
        # Resize the initial mask to match the new slice and simulate growth
        self.tumor_mask_resized_sagittal = self.resize_mask_to_slice(self.mri_data['glistrboost'][slice_idx, :, :] > 0,
                                                                         self.mri_data[cur_scan].shape[1:])
        self.tumor_mask_resized_coronal = self.resize_mask_to_slice(self.mri_data['glistrboost'][:, slice_idx, :] > 0,
                                                                        self.mri_data[cur_scan].shape[1:])
        self.tumor_mask_resized_axial = self.resize_mask_to_slice(self.mri_data['glistrboost'][:, :, slice_idx] > 0,
                                                                      self.mri_data[cur_scan].shape[:2])
        # Prepare new second segmentation masks
        self.second_segmentation_mask_resized_sagittal = self.resize_mask_to_slice(
            self.mri_data['seg2'][slice_idx, :, :] > 0,
            self.mri_data[cur_scan].shape[1:]
        )
        self.second_segmentation_mask_resized_coronal = self.resize_mask_to_slice(
            self.mri_data['seg2'][:, slice_idx, :] > 0,
            self.mri_data[cur_scan].shape[1:]
        )
        self.second_segmentation_mask_resized_axial = self.resize_mask_to_slice(
            self.mri_data['seg2'][:, :, slice_idx] > 0,
            self.mri_data[cur_scan].shape[:2]
        )

        try:
            # Extract the diffusion map slice dynamically
            self.diffusion_map_sagittal = self.diffusion_map[slice_idx, :, :]  # sagittal
            self.diffusion_map_coronal = self.diffusion_map[:, slice_idx, :]  # coronal
            self.diffusion_map_axial = self.diffusion_map[:, :, slice_idx]  # axial

            self.grown_tumor_mask_sagittal = self.simulate_growth(self.tumor_mask_resized_sagittal,
                                                                  diffusion_rate=self.diffusion_map_sagittal,
                                                                  reaction_rate=self.reaction_rate,
                                                                  time_steps=time_step,
                                                                  brain_mask=self.brain_mask_sagittal)
            self.grown_tumor_mask_coronal = self.simulate_growth(self.tumor_mask_resized_coronal,
                                                                 diffusion_rate=self.diffusion_map_coronal,
                                                                 reaction_rate=self.reaction_rate, time_steps=time_step,
                                                                 brain_mask=self.brain_mask_coronal)
            self.grown_tumor_mask_axial = self.simulate_growth(self.tumor_mask_resized_axial,
                                                               diffusion_rate=self.diffusion_map_axial,
                                                               reaction_rate=self.reaction_rate, time_steps=time_step,
                                                               brain_mask=self.brain_mask_axial)
        except:
            self.grown_tumor_mask_sagittal = self.simulate_growth(self.tumor_mask_resized_sagittal,
                                                                  diffusion_rate=self.diffusion_rate,
                                                                  reaction_rate=self.reaction_rate,
                                                                  time_steps=time_step,
                                                                  brain_mask=self.brain_mask_sagittal)
            self.grown_tumor_mask_coronal = self.simulate_growth(self.tumor_mask_resized_coronal,
                                                                 diffusion_rate=self.diffusion_rate,
                                                                 reaction_rate=self.reaction_rate, time_steps=time_step,
                                                                 brain_mask=self.brain_mask_coronal)
            self.grown_tumor_mask_axial = self.simulate_growth(self.tumor_mask_resized_axial,
                                                               diffusion_rate=self.diffusion_rate,
                                                               reaction_rate=self.reaction_rate, time_steps=time_step,
                                                               brain_mask=self.brain_mask_axial)

        # Apply tumor overlays
        if overlay:
            self.scan_rgb_sagittal[self.grown_tumor_mask_sagittal.T, 0] = 1
            self.scan_rgb_sagittal[self.grown_tumor_mask_sagittal.T, 1] = 0
            self.scan_rgb_sagittal[self.grown_tumor_mask_sagittal.T, 2] = 0

            self.scan_rgb_coronal[self.grown_tumor_mask_coronal.T, 0] = 1
            self.scan_rgb_coronal[self.grown_tumor_mask_coronal.T, 1] = 0
            self.scan_rgb_coronal[self.grown_tumor_mask_coronal.T, 2] = 0

            self.scan_rgb_axial[self.grown_tumor_mask_axial.T, 0] = 1
            self.scan_rgb_axial[self.grown_tumor_mask_axial.T, 1] = 0
            self.scan_rgb_axial[self.grown_tumor_mask_axial.T, 2] = 0

            # Apply second segmentation overlays in green
            self.scan_rgb_sagittal[self.second_segmentation_mask_resized_sagittal.T, 0] = 0
            self.scan_rgb_sagittal[self.second_segmentation_mask_resized_sagittal.T, 1] = 1
            self.scan_rgb_sagittal[self.second_segmentation_mask_resized_sagittal.T, 2] = 0

            self.scan_rgb_coronal[self.second_segmentation_mask_resized_coronal.T, 0] = 0
            self.scan_rgb_coronal[self.second_segmentation_mask_resized_coronal.T, 1] = 1
            self.scan_rgb_coronal[self.second_segmentation_mask_resized_coronal.T, 2] = 0

            self.scan_rgb_axial[self.second_segmentation_mask_resized_axial.T, 0] = 0
            self.scan_rgb_axial[self.second_segmentation_mask_resized_axial.T, 1] = 1
            self.scan_rgb_axial[self.second_segmentation_mask_resized_axial.T, 2] = 0

        # Update the images with the new slice and tumor mask
        self.scan_img_sagittal.set_data(self.scan_rgb_sagittal)
        self.scan_img_coronal.set_data(self.scan_rgb_coronal)
        self.scan_img_axial.set_data(self.scan_rgb_axial)

        self.fig.canvas.draw_idle()

    def get_max_slice_value(self, mri_data, current_scan):
        mri_shape = mri_data[current_scan].shape
        num_slices_sagittal = mri_shape[0]
        num_slices_coronal = mri_shape[1]
        num_slices_axial = mri_shape[2]

        min_slices = min(num_slices_sagittal, num_slices_coronal, num_slices_axial)
        return min_slices

    def handle_args(self):
        parser = argparse.ArgumentParser(description="Choose how to load files.")
        parser.add_argument(
            '-a',
            '--auto',
            action='store_true',
            help="Automatically load files by searching the current directory and subdirectories."
        )
        args = parser.parse_args()
        return args

    def run_ants_diffusion_map(self, t1_file, grey_d, white_d, diffusion):
        # Run antDiffusionMap in a subprocess
        print("start run another ants")
        subprocess.run([sys.executable, "antDiffusionMap.py", t1_file, str(grey_d), str(white_d), str(diffusion)], check=True)
        print("after subprocess run")

    def start_equation(self, cur_scan, grey_diffusion, white_diffusion):

        self.grey_diffusion_rate = grey_diffusion
        self.white_diffusion_rate = white_diffusion

        self.mri_data = self.load_mri_data(self.file_paths)  # Load the MRI data

        max_slices = self.get_max_slice_value(self.mri_data, cur_scan) - 1

        # ====================== Generate Diffusion Map =========================================
        process = multiprocessing.Process(target=self.run_ants_diffusion_map, args=(self.file_paths["t1"],
                                                                                    grey_diffusion, white_diffusion,
                                                                                    self.csf_diffusion_rate))

        process.start()
        print("process start")
        # process.join(timeout=180)
        process.join()

        print("finish diffusion map")

        map = np.load('diffusion_map.npy')  # load diffusion map as numpy

        if map is not None:
            print(f"has diffusion mask: {map}")
        else:
            print(f"error diffusion mask: {map}")
            map = []
        # ======================= End of Generate Diffusion Map ====================================

        self.diffusion_map = np.where( map> 0, map, EquationConstant.DIFFUSION_RATE)

        testFig = self.interactive_growth_visualization_2(self.mri_data, cur_scan)
        cur_slice_index = self.sagittal_slice_idx
        print("finish start equation")
        return testFig, cur_slice_index, max_slices

    def update_file_paths(self, path_key, path_value):
        self.file_paths[path_key] = path_value

    def save_tumor_mask_as_nii(self, tumor_mask, reference_nii_path, output_path="grown_tumor_mask.nii"):
        # Load the reference NIfTI image to get its affine matrix
        reference_img = nib.load(reference_nii_path)
        original_affine = reference_img.affine # Get the original affine matrix
        scale_factors = np.array(tumor_mask.shape) / np.array(reference_img.shape[:3]) # Calculate the scaling factors
        
        new_affine = original_affine.copy() # Create a new affine matrix with adjusted scaling
        new_affine[:3, :3] *= scale_factors[:, np.newaxis]  
        # Create and save the NIfTI image with the new affine matrix
        tumor_img = nib.Nifti1Image(tumor_mask.astype(np.uint8), new_affine)
        nib.save(tumor_img, output_path)
        print(f"\nGrown tumor mask saved as {output_path}")

        # Debugging: Load the saved mask and verify its properties
        saved_mask_img = nib.load(output_path)
        print("\nDebugging Saved Mask Properties:")
        print("Saved mask affine matrix:\n", saved_mask_img.affine)
        print("Saved mask data shape:", saved_mask_img.get_fdata().shape)
        print("Saved mask unique values:", np.unique(saved_mask_img.get_fdata()))

    def save_current_mask(self, s_index, t_index):
        slice_idx = int(s_index)
        time_step = int(t_index)

        # Simulate tumor growth for the entire 3D volume
        full_tumor_mask = np.zeros_like(self.mri_data['flair'], dtype=bool)  # Initialize a 3D mask

        for i in range(self.mri_data['flair'].shape[0]):  # Iterate over all slices
            # Simulate growth for each slice
            slice_mask = self.simulate_growth(
                self.resize_mask_to_slice(self.mri_data['glistrboost'][i, :, :] > 0, self.mri_data['flair'].shape[1:]),
                diffusion_rate=self.diffusion_map[i, :, :],
                reaction_rate=self.reaction_rate,
                time_steps=time_step,
                brain_mask=self.create_brain_mask(self.mri_data['flair'][i, :, :])
            )
            full_tumor_mask[i, :, :] = slice_mask  # Add the slice to the 3D mask
        first_segmentation_name = os.path.basename(self.file_paths['glistrboost'])
        first_segmentation_name = os.path.splitext(first_segmentation_name)[0]
        current_date = datetime.now().strftime("%Y%m%d")
        output_filename = f"{first_segmentation_name}_grown_{current_date}.nii"
        
        # Debugging: Print the shape of the full 3D mask
        print("\nDebugging Full 3D Tumor Mask Shape:")
        print("Full tumor mask shape:", full_tumor_mask.shape)

        # Save the full 3D mask using the FLAIR image as the reference
        # output_path = f"grown_tumor_mask_time_{time_step}.nii"
        output_path = os.path.join(os.path.dirname(self.file_paths['flair']), output_filename)
        reference_nii_path = self.file_paths['flair']  # Use FLAIR as the reference image
        self.save_tumor_mask_as_nii(full_tumor_mask, reference_nii_path, output_path)

if __name__ == "__main__":
    obj = BiologicalModel.instance()
    obj.without_app = True
    args = obj.handle_args()

    if args.auto:
        print("Generating model with auto-selected files...")
        file_paths = obj.auto_load_files()
    else:
        file_paths = obj.get_file_paths()

    mri_data = obj.load_mri_data(file_paths) # Load the MRI data

    # Initialize the interactive visualization
    # obj.start_equation()
    # Initialize the interactive visualization
    obj.interactive_growth_visualization_2(mri_data, cur_scan = 'flair')
