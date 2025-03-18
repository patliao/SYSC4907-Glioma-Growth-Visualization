import argparse
import os, sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import platform
from nipype import Workflow, Node
import subprocess
import platform
import matplotlib
import matplotlib.pyplot as plt
if platform.system() == "Darwin":
    matplotlib.use("Qt5Agg")
import nibabel as nib
import numpy as np
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from Application.equation_constant import EquationConstant
from scipy.ndimage import distance_transform_edt, center_of_mass
import ants

matplotlib.use('TkAgg')

class BiologicalModel:
    _instance = None
    
    def __init__(self):
        self.file_paths = {}
        self.diffusion_rate = EquationConstant.DIFFUSION_RATE
        self.reaction_rate = EquationConstant.REACTION_RATE

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = BiologicalModel()
        return cls._instance
    
    def load_second_segmentation(self, file_path):
        """Load the second segmentation mask."""
        return nib.load(file_path).get_fdata()
    
    def set_diffusion_rate(self, diffusion_rate):
        self.diffusion_rate = diffusion_rate

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

    def load_mri_data(self, file_paths):
        """Step 1: Load MRI Data"""
        return {key: nib.load(file).get_fdata() for key, file in file_paths.items()}

    def resize_mask_to_slice(self, tumor_mask, slice_shape, dtype=bool):
        """Step 2: Resize the tumor mask to match the slice shape."""
        resized_mask = resize(
            tumor_mask,
            slice_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )
        return resized_mask.astype(dtype)

    def simulate_growth(self, initial_mask, diffusion_rate, time_steps, brain_mask):
        """Step 3: Simulate Tumor Growth using Reaction-Diffusion"""
        mask = initial_mask.copy().astype(float)
        brain_mask_resized = self.resize_mask_to_slice(brain_mask, mask.shape)
        diffusion_map_resized = self.resize_mask_to_slice(diffusion_rate, mask.shape, dtype=float)
        decay = self.get_decay_factor(mask)

        for _ in range(time_steps):
            diffused_mask = gaussian_filter(mask, sigma=1.0) * diffusion_map_resized * decay
            growth = self.reaction_rate * mask * (1 - mask) * decay

            mask = brain_mask_resized * (mask + diffused_mask + growth)
            mask = np.clip(mask, 0, 1)  # keep values in [0, 1]

        return mask > 0.5
    
    def get_decay_factor(self, mask):
        distance_map = distance_transform_edt(mask == 0)
        decay_factor = np.exp(-distance_map / EquationConstant.LAMBDA)
        return decay_factor

    def save_tumor_mask_as_nii(self, tumor_mask, reference_nii_path, output_path="grown_tumor_mask.nii"):
        """Save the grown tumor mask as a .nii file using the affine matrix from the reference image."""
        # Debugging: Print mask properties before saving
        print("\nDebugging Mask Properties:")
        print("Mask shape:", tumor_mask.shape)
        print("Mask unique values:", np.unique(tumor_mask))
        print("Mask non-zero voxels:", np.count_nonzero(tumor_mask))

        # Load the reference NIfTI image to get its affine matrix
        reference_img = nib.load(reference_nii_path)
        print("\nDebugging Reference Image Properties:")
        print("Reference image shape:", reference_img.shape)
        print("Reference affine matrix:\n", reference_img.affine)

        # Ensure the mask is in the correct shape and data type
        if tumor_mask.shape != reference_img.shape:
            raise ValueError(f"Mask shape {tumor_mask.shape} does not match reference shape {reference_img.shape}.")

        # Convert the binary mask to uint8 (0 and 1)
        tumor_mask_int = tumor_mask.astype(np.uint8)

        # Debugging: Print the mask data type and values after conversion
        print("\nDebugging Mask After Conversion:")
        print("Mask data type:", tumor_mask_int.dtype)
        print("Mask unique values after conversion:", np.unique(tumor_mask_int))

        # Create and save the NIfTI image
        tumor_img = nib.Nifti1Image(tumor_mask_int, reference_img.affine)
        nib.save(tumor_img, output_path)
        print(f"\nGrown tumor mask saved as {output_path}")

        # Debugging: Load the saved mask and verify its properties
        saved_mask_img = nib.load(output_path)
        print("\nDebugging Saved Mask Properties:")
        print("Saved mask affine matrix:\n", saved_mask_img.affine)
        print("Saved mask data shape:", saved_mask_img.get_fdata().shape)
        print("Saved mask unique values:", np.unique(saved_mask_img.get_fdata()))

    def interactive_growth_visualization(self, mri_data, diffusion_map):
        """Step 4: Interactive Visualization with Slice, Time Sliders, and Overlay Toggle"""
        sagittal_slice_idx = mri_data['flair'].shape[0] // 2
        coronal_slice_idx = mri_data['flair'].shape[1] // 2
        axial_slice_idx = mri_data['flair'].shape[2] // 2

        # Get the initial tumor mask for sagittal, coronal, axial slices
        initial_tumor_mask_sagittal = mri_data['glistrboost'][sagittal_slice_idx, :, :] > 0
        initial_tumor_mask_coronal = mri_data['glistrboost'][:, coronal_slice_idx, :] > 0
        initial_tumor_mask_axial = mri_data['glistrboost'][:, :, axial_slice_idx] > 0

        second_segmentation_mask_sagittal = mri_data['seg2'][sagittal_slice_idx, :, :] > 0
        second_segmentation_mask_coronal = mri_data['seg2'][:, coronal_slice_idx, :] > 0
        second_segmentation_mask_axial = mri_data['seg2'][:, :, axial_slice_idx] > 0
    # Debug: Print shapes and unique values of the second segmentation mask
        print("\nDebugging Second Segmentation Mask:")
        print(f"Sagittal slice shape: {second_segmentation_mask_sagittal.shape}, unique values: {np.unique(second_segmentation_mask_sagittal)}")
        print(f"Coronal slice shape: {second_segmentation_mask_coronal.shape}, unique values: {np.unique(second_segmentation_mask_coronal)}")
        print(f"Axial slice shape: {second_segmentation_mask_axial.shape}, unique values: {np.unique(second_segmentation_mask_axial)}")
        # Resize the tumor masks
        tumor_mask_resized_sagittal = self.resize_mask_to_slice(
            initial_tumor_mask_sagittal,
            mri_data['flair'].shape[1:3]
        )
        tumor_mask_resized_coronal = self.resize_mask_to_slice(
            initial_tumor_mask_coronal,
            mri_data['flair'].shape[1:3]
        )
        tumor_mask_resized_axial = self.resize_mask_to_slice(
            initial_tumor_mask_axial,
            mri_data['flair'].shape[:2]
        )

        # Resize the second segmentation masks
        second_segmentation_mask_resized_sagittal = self.resize_mask_to_slice(
            second_segmentation_mask_sagittal,
            mri_data['flair'].shape[1:3]
        )
        second_segmentation_mask_resized_coronal = self.resize_mask_to_slice(
            second_segmentation_mask_coronal,
            mri_data['flair'].shape[1:3]
        )
        second_segmentation_mask_resized_axial = self.resize_mask_to_slice(
            second_segmentation_mask_axial,
            mri_data['flair'].shape[:2]
        )
        print("\nDebugging Resized Second Segmentation Mask:")
        print(f"Sagittal slice shape: {second_segmentation_mask_resized_sagittal.shape}, unique values: {np.unique(second_segmentation_mask_resized_sagittal)}")
        print(f"Coronal slice shape: {second_segmentation_mask_resized_coronal.shape}, unique values: {np.unique(second_segmentation_mask_resized_coronal)}")
        print(f"Axial slice shape: {second_segmentation_mask_resized_axial.shape}, unique values: {np.unique(second_segmentation_mask_resized_axial)}")
        fig, (ax_sagittal, ax_coronal, ax_axial) = plt.subplots(1, 3, figsize=(14, 7))
        plt.subplots_adjust(left=0.25, bottom=0.35)

        # Start with the 'flair' scan
        current_scan = 'flair'
        scan_slice_sagittal = mri_data[current_scan][sagittal_slice_idx, :, :].T
        scan_slice_coronal = mri_data[current_scan][:, coronal_slice_idx, :].T
        scan_slice_axial = mri_data[current_scan][:, :, axial_slice_idx].T

        # Convert to RGB
        scan_rgb_sagittal = np.repeat(scan_slice_sagittal[:, :, np.newaxis], 3, axis=2)
        scan_rgb_coronal = np.repeat(scan_slice_coronal[:, :, np.newaxis], 3, axis=2)
        scan_rgb_axial = np.repeat(scan_slice_axial[:, :, np.newaxis], 3, axis=2)

        # Normalize
        scan_rgb_sagittal = np.clip(scan_rgb_sagittal / np.max(scan_rgb_sagittal), 0, 1)
        scan_rgb_coronal = np.clip(scan_rgb_coronal / np.max(scan_rgb_coronal), 0, 1)
        scan_rgb_axial = np.clip(scan_rgb_axial / np.max(scan_rgb_axial), 0, 1)

        # Display the slices
        scan_img_sagittal = ax_sagittal.imshow(scan_rgb_sagittal, origin='lower')
        scan_img_coronal = ax_coronal.imshow(scan_rgb_coronal, origin='lower')
        scan_img_axial = ax_axial.imshow(scan_rgb_axial, origin='lower')

        # Prepare the overlay
        overlay_on = True
        tumor_overlay_sagittal = tumor_mask_resized_sagittal.T
        tumor_overlay_coronal = tumor_mask_resized_coronal.T
        tumor_overlay_axial = tumor_mask_resized_axial.T

        second_seg_overlay_sagittal = second_segmentation_mask_resized_sagittal.T
        second_seg_overlay_coronal = second_segmentation_mask_resized_coronal.T
        second_seg_overlay_axial = second_segmentation_mask_resized_axial.T

        # Apply tumor overlays in red
        scan_rgb_sagittal[tumor_overlay_sagittal, 0] = 1
        scan_rgb_sagittal[tumor_overlay_sagittal, 1] = 0
        scan_rgb_sagittal[tumor_overlay_sagittal, 2] = 0

        # Apply second segmentation overlays in green
        scan_rgb_sagittal[second_seg_overlay_sagittal, 0] = 0
        scan_rgb_sagittal[second_seg_overlay_sagittal, 1] = 1
        scan_rgb_sagittal[second_seg_overlay_sagittal, 2] = 0

        scan_rgb_coronal[tumor_overlay_coronal, 0] = 1
        scan_rgb_coronal[tumor_overlay_coronal, 1] = 0
        scan_rgb_coronal[tumor_overlay_coronal, 2] = 0

        scan_rgb_coronal[second_seg_overlay_coronal, 0] = 0
        scan_rgb_coronal[second_seg_overlay_coronal, 1] = 1
        scan_rgb_coronal[second_seg_overlay_coronal, 2] = 0

        scan_rgb_axial[tumor_overlay_axial, 0] = 1
        scan_rgb_axial[tumor_overlay_axial, 1] = 0
        scan_rgb_axial[tumor_overlay_axial, 2] = 0

        scan_rgb_axial[second_seg_overlay_axial, 0] = 0
        scan_rgb_axial[second_seg_overlay_axial, 1] = 1
        scan_rgb_axial[second_seg_overlay_axial, 2] = 0

        scan_img_sagittal.set_data(scan_rgb_sagittal)
        scan_img_coronal.set_data(scan_rgb_coronal)
        scan_img_axial.set_data(scan_rgb_axial)

        # Slider for slice index
        ax_slice_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
        min_slices = self.get_max_slice_value(mri_data, current_scan)
        slice_slider = Slider(
            ax_slice_slider,
            'Slice Index',
            0,
            min_slices - 1,
            valinit=sagittal_slice_idx,
            valstep=1
        )

        # Slider for time steps
        ax_time_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        time_slider = Slider(
            ax_time_slider,
            'Time Step',
            0,
            EquationConstant.NUM_STEPS,
            valinit=0,
            valstep=1
        )

        def calculate_time_in_days(step):
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

        def update_time_step(val):
            calculated_time = calculate_time_in_days(val)
            time_slider.valtext.set_text(f"{calculated_time:.2f} days")

        time_slider.on_changed(update_time_step)

        # Local create_brain_mask for demonstration
        def create_brain_mask(mri_image):
            return mri_image > 0

                # -------- Save Button --------
        def save_current_mask(event):
            slice_idx = int(slice_slider.val)
            time_step = int(time_slider.val)

            # Simulate tumor growth for the entire 3D volume
            full_tumor_mask = np.zeros_like(mri_data['flair'], dtype=bool)  # Initialize a 3D mask

            for i in range(mri_data['flair'].shape[0]):  # Iterate over all slices
                # Simulate growth for each slice
                slice_mask = self.simulate_growth(
                    self.resize_mask_to_slice(mri_data['glistrboost'][i, :, :] > 0, mri_data['flair'].shape[1:]),
                    diffusion_rate=diffusion_map[i, :, :],
                    time_steps=time_step,
                    brain_mask=create_brain_mask(mri_data['flair'][i, :, :])
                )
                full_tumor_mask[i, :, :] = slice_mask  # Add the slice to the 3D mask

            # Debugging: Print the shape of the full 3D mask
            print("\nDebugging Full 3D Tumor Mask Shape:")
            print("Full tumor mask shape:", full_tumor_mask.shape)

            # Save the full 3D mask using the FLAIR image as the reference
            output_path = f"grown_tumor_mask_time_{time_step}.nii"
            reference_nii_path = self.file_paths['flair']  # Use FLAIR as the reference image
            self.save_tumor_mask_as_nii(full_tumor_mask, reference_nii_path, output_path)
        ax_save_button = plt.axes([0.05, 0.3, 0.15, 0.05])
        save_button = Button(ax_save_button, 'Save Mask')
        save_button.on_clicked(save_current_mask)

        # -------- Overlay Toggle --------
        ax_toggle = plt.axes([0.05, 0.5, 0.15, 0.15])
        toggle_button = CheckButtons(ax_toggle, ['Toggle Overlay'], [overlay_on])
        for label in toggle_button.labels:
            label.set_fontsize(10)
            label.set_color('black')

        ax_toggle.spines['top'].set_visible(False)
        ax_toggle.spines['right'].set_visible(False)
        ax_toggle.spines['left'].set_visible(False)
        ax_toggle.spines['bottom'].set_visible(False)

        def toggle_overlay(label):
            nonlocal overlay_on
            overlay_on = not overlay_on
            update(None)

        toggle_button.on_clicked(toggle_overlay)

        # -------- Radio Buttons for Scan Types --------
        ax_radio = plt.axes([0.05, 0.8, 0.15, 0.15])
        radio_button = RadioButtons(ax_radio, ['FLAIR', 'T1', 'T1 GD', 'T2'])

        def update_scan_type(label):
            nonlocal current_scan
            current_scan = label.lower()  # 'T1 GD' => 't1 gd' => 't1gd' if you want
            update(None)

        radio_button.on_clicked(update_scan_type)

        def update(val):
            slice_idx = int(slice_slider.val)
            time_step = int(time_slider.val)

            # Local create_brain_mask
            def create_brain_mask(mri_image):
                return mri_image > 0

            brain_mask_sagittal = create_brain_mask(mri_data['flair'][slice_idx, :, :])
            brain_mask_coronal = create_brain_mask(mri_data['flair'][:, slice_idx, :])
            brain_mask_axial = create_brain_mask(mri_data['flair'][:, :, slice_idx])

            # Get diffusion slices
            diffusion_map_sagittal = diffusion_map[slice_idx, :, :]
            diffusion_map_coronal = diffusion_map[:, slice_idx, :]
            diffusion_map_axial = diffusion_map[:, :, slice_idx]

            # Resize brain masks
            brain_mask_resized_sagittal = self.resize_mask_to_slice(
                brain_mask_sagittal,
                mri_data[current_scan].shape[1:]
            )
            brain_mask_resized_coronal = self.resize_mask_to_slice(
                brain_mask_coronal,
                mri_data[current_scan].shape[1:]
            )
            brain_mask_resized_axial = self.resize_mask_to_slice(
                brain_mask_axial,
                mri_data[current_scan].shape[:2]
            )

            # Update the selected slice
            scan_slice_sagittal = mri_data[current_scan][slice_idx, :, :].T
            scan_slice_coronal = mri_data[current_scan][:, slice_idx, :].T
            scan_slice_axial = mri_data[current_scan][:, :, slice_idx].T

            scan_rgb_sagittal = np.repeat(scan_slice_sagittal[:, :, np.newaxis], 3, axis=2)
            scan_rgb_coronal = np.repeat(scan_slice_coronal[:, :, np.newaxis], 3, axis=2)
            scan_rgb_axial = np.repeat(scan_slice_axial[:, :, np.newaxis], 3, axis=2)

            scan_rgb_sagittal = np.clip(scan_rgb_sagittal / np.max(scan_rgb_sagittal), 0, 1)
            scan_rgb_coronal = np.clip(scan_rgb_coronal / np.max(scan_rgb_coronal), 0, 1)
            scan_rgb_axial = np.clip(scan_rgb_axial / np.max(scan_rgb_axial), 0, 1)

            # Prepare new tumor masks
            tumor_mask_resized_sagittal = self.resize_mask_to_slice(
                mri_data['glistrboost'][slice_idx, :, :] > 0,
                mri_data[current_scan].shape[1:]
            )
            tumor_mask_resized_coronal = self.resize_mask_to_slice(
                mri_data['glistrboost'][:, slice_idx, :] > 0,
                mri_data[current_scan].shape[1:]
            )
            tumor_mask_resized_axial = self.resize_mask_to_slice(
                mri_data['glistrboost'][:, :, slice_idx] > 0,
                mri_data[current_scan].shape[:2]
            )

            # Prepare new second segmentation masks
            second_segmentation_mask_resized_sagittal = self.resize_mask_to_slice(
                mri_data['seg2'][slice_idx, :, :] > 0,
                mri_data[current_scan].shape[1:]
            )
            second_segmentation_mask_resized_coronal = self.resize_mask_to_slice(
                mri_data['seg2'][:, slice_idx, :] > 0,
                mri_data[current_scan].shape[1:]
            )
            second_segmentation_mask_resized_axial = self.resize_mask_to_slice(
                mri_data['seg2'][:, :, slice_idx] > 0,
                mri_data[current_scan].shape[:2]
            )

            # Simulate growth
            grown_tumor_mask_sagittal = self.simulate_growth(
                tumor_mask_resized_sagittal,
                diffusion_rate=diffusion_map_sagittal,
                time_steps=time_step,
                brain_mask=brain_mask_sagittal
            )
            grown_tumor_mask_coronal = self.simulate_growth(
                tumor_mask_resized_coronal,
                diffusion_rate=diffusion_map_coronal,
                time_steps=time_step,
                brain_mask=brain_mask_coronal
            )
            grown_tumor_mask_axial = self.simulate_growth(
                tumor_mask_resized_axial,
                diffusion_rate=diffusion_map_axial,
                time_steps=time_step,
                brain_mask=brain_mask_axial
            )

            if overlay_on:
                # Apply tumor overlays in red
                scan_rgb_sagittal[grown_tumor_mask_sagittal.T, 0] = 1
                scan_rgb_sagittal[grown_tumor_mask_sagittal.T, 1] = 0
                scan_rgb_sagittal[grown_tumor_mask_sagittal.T, 2] = 0

                scan_rgb_coronal[grown_tumor_mask_coronal.T, 0] = 1
                scan_rgb_coronal[grown_tumor_mask_coronal.T, 1] = 0
                scan_rgb_coronal[grown_tumor_mask_coronal.T, 2] = 0

                scan_rgb_axial[grown_tumor_mask_axial.T, 0] = 1
                scan_rgb_axial[grown_tumor_mask_axial.T, 1] = 0
                scan_rgb_axial[grown_tumor_mask_axial.T, 2] = 0

                # Apply second segmentation overlays in green
                scan_rgb_sagittal[second_segmentation_mask_resized_sagittal.T, 0] = 0
                scan_rgb_sagittal[second_segmentation_mask_resized_sagittal.T, 1] = 1
                scan_rgb_sagittal[second_segmentation_mask_resized_sagittal.T, 2] = 0

                scan_rgb_coronal[second_segmentation_mask_resized_coronal.T, 0] = 0
                scan_rgb_coronal[second_segmentation_mask_resized_coronal.T, 1] = 1
                scan_rgb_coronal[second_segmentation_mask_resized_coronal.T, 2] = 0

                scan_rgb_axial[second_segmentation_mask_resized_axial.T, 0] = 0
                scan_rgb_axial[second_segmentation_mask_resized_axial.T, 1] = 1
                scan_rgb_axial[second_segmentation_mask_resized_axial.T, 2] = 0

            # Update the images
            scan_img_sagittal.set_data(scan_rgb_sagittal)
            scan_img_coronal.set_data(scan_rgb_coronal)
            scan_img_axial.set_data(scan_rgb_axial)

            fig.canvas.draw_idle()
        # Link update to sliders
        slice_slider.on_changed(update)
        time_slider.on_changed(update)

        # Set background color
        ax_sagittal.set_facecolor('black')
        ax_coronal.set_facecolor('black')
        ax_axial.set_facecolor('black')

        plt.show()
        return fig

    def get_max_slice_value(self, mri_data, current_scan):
        mri_shape = mri_data[current_scan].shape
        num_slices_sagittal = mri_shape[0]
        num_slices_coronal = mri_shape[1]
        num_slices_axial = mri_shape[2]
        return min(num_slices_sagittal, num_slices_coronal, num_slices_axial)

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

    def start_equation(self):
        """ Main driver function to load data, create diffusion map, and start visualization. """
        mri_data = self.load_mri_data(self.file_paths)
        initial_diffusion_map = self.create_diffusion_map(self.file_paths["t1"])
        diffusion_map = np.where(initial_diffusion_map > 0, initial_diffusion_map, EquationConstant.DIFFUSION_RATE)
        seg2_path = r"C:\Users\Frankii Siconolfi\YEARFOUR\SYSC4907-Glioma-Growth-Visualization\100011\100011_time2_seg.nii.gz"
        mri_data['seg2'] = nib.load(seg2_path).get_fdata()
        fig = self.interactive_growth_visualization(mri_data, diffusion_map)
        return fig

    def update_file_paths(self, path_key, path_value):
        self.file_paths[path_key] = path_value

    def create_diffusion_map(self, t1_image):
        threshold = 0.5
        print("Segmenting MRI data (this will take several moments)...")

        t1_image_path = r"{}".format(t1_image)
        t1_image = ants.image_read(t1_image_path)
        t1_corrected = ants.n4_bias_field_correction(t1_image)
        t1_normalized = ants.iMath(t1_corrected, "Normalize")

        brain_mask = ants.get_mask(t1_normalized)
        refined_mask = ants.iMath(brain_mask, "MD", 2)

        segmentation = ants.atropos(
            a=t1_normalized,
            x=refined_mask,
            i=f'kmeans[5]',
            m='[0.6,1x1x1]',
            c='[10,0.01]'
        )

        csf_map = segmentation['probabilityimages'][0] + segmentation['probabilityimages'][1]  # CSF
        gm_map = segmentation['probabilityimages'][2] + segmentation['probabilityimages'][3]  # GM
        wm_map = segmentation['probabilityimages'][4]  # WM

        csf_map = ants.threshold_image(csf_map, threshold, 1)
        gm_map = ants.threshold_image(gm_map, threshold, 1)
        wm_map = ants.threshold_image(wm_map, threshold, 1)

        csf_data = csf_map.numpy()
        gm_data = gm_map.numpy()
        wm_data = wm_map.numpy()

        diffusion_map = np.zeros_like(gm_data)
        diffusion_map[csf_data > 0] = EquationConstant.CSF_DIFFUSION_RATE
        diffusion_map[gm_data > 0] = EquationConstant.GREY_DIFFUSION_RATE
        diffusion_map[wm_data > 0] = EquationConstant.WHITE_DIFFUSION_RATE

        return diffusion_map

if __name__ == "__main__":
    obj = BiologicalModel.instance()
    args = obj.handle_args()

    if args.auto:
        print("Generating model with auto-selected files...")
        file_paths = obj.auto_load_files()
    else:
        file_paths = obj.get_file_paths()

    obj.file_paths = file_paths
    # Run the simulation
    obj.start_equation()
