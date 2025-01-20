import argparse
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from Application.equation_constant import EquationConstant

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

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = BiologicalModel()
        return cls._instance

    def get_diffusion_rate(self):
        return self.diffusion_rate

    def set_diffusion_rate(self, diffusion_rate):
        self.diffusion_rate = diffusion_rate

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

        return file_paths

    # Step 1: Load MRI Data
    def load_mri_data(self, file_paths):
        return {key: nib.load(file).get_fdata() for key, file in file_paths.items()}


    # Step 2: Resize the tumor mask to match the slice shape
    def resize_mask_to_slice(self, tumor_mask, slice_shape):
        resized_mask = resize(tumor_mask, slice_shape, order=0, preserve_range=True, anti_aliasing=False)
        return resized_mask.astype(bool)

    # Step 3: Simulate Tumor Growth using Reaction-Diffusion
    def simulate_growth(self, initial_mask, diffusion_rate, reaction_rate, time_steps, brain_mask):
        mask = initial_mask.copy().astype(float)
        brain_mask_resized = self.resize_mask_to_slice(brain_mask, mask.shape)
        for _ in range(time_steps):
            # Apply Gaussian filter for diffusion and add reaction (growth)
            mask = gaussian_filter(mask, sigma=diffusion_rate)
            growth = reaction_rate * mask * (1 - mask)
            mask = mask + (growth * brain_mask_resized) # ensure the growth is kept within the brain region
            mask = np.clip(mask, 0, 1)  # Keep values in range

        return mask > 0.5  # Threshold to keep mask as binary

    def time_in_days(self, step):
        time_step = (EquationConstant.SPATIAL_RESOLUTION ** 2) / (2 * 3 * self.diffusion_rate)
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
        # Set up the figure and axis
        fig, (self.ax_sagittal, self.ax_coronal, self.ax_axial) = plt.subplots(1, 3, figsize=(14, 7))
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

        # Apply tumor overlays
        self.scan_rgb_sagittal[self.tumor_overlay_sagittal, 0] = 1
        self.scan_rgb_sagittal[self.tumor_overlay_sagittal, 1] = 0
        self.scan_rgb_sagittal[self.tumor_overlay_sagittal, 2] = 0

        self.scan_rgb_coronal[self.tumor_overlay_coronal, 0] = 1
        self.scan_rgb_coronal[self.tumor_overlay_coronal, 1] = 0
        self.scan_rgb_coronal[self.tumor_overlay_coronal, 2] = 0

        self.scan_rgb_axial[self.tumor_overlay_axial, 0] = 1
        self.scan_rgb_axial[self.tumor_overlay_axial, 1] = 0
        self.scan_rgb_axial[self.tumor_overlay_axial, 2] = 0

        self.scan_img_sagittal.set_data(self.scan_rgb_sagittal)
        self.scan_img_coronal.set_data(self.scan_rgb_coronal)
        self.scan_img_axial.set_data(self.scan_rgb_axial)

        # self.ax_slice_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
        # self.min_slices = self.get_max_slice_value(mri_data, self.current_scan)
        # self.slice_slider = Slider(self.ax_slice_slider, 'Slice Index', 0, self.min_slices - 1, valinit=self.sagittal_slice_idx, valstep=1)

        # # Slider for controlling time step
        # self.ax_time_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        # self.time_slider = Slider(self.ax_time_slider, 'Time Step', 0, EquationConstant.NUM_STEPS, valinit=0, valstep=1)

        # def update_time_step(val):
        #     self.calculated_time = calculate_time_in_days(val)
        #     self.time_slider.valtext.set_text(f"{calculated_time:.2f} days")
        #


        # self.time_slider.on_changed(update_time_step)

        # # Checkbox for toggling red overlay
        # self.ax_toggle = plt.axes([0.05, 0.5, 0.15, 0.15])
        # self.toggle_button = self.CheckButtons(ax_toggle, ['Toggle Overlay'], [overlay_on])

        # # RadioButtons for selecting scan type (FLAIR, T1, T1_Gd, etc.)
        # self.ax_radio = plt.axes([0.05, 0.8, 0.15, 0.15])
        # self.radio_button = RadioButtons(self.ax_radio, ['FLAIR', 'T1', 'T1 GD', 'T2'])

        # # Hide the border around the checkbox
        # for label in self.toggle_button.labels:
        #     label.set_fontsize(10)
        #     label.set_color('black')

        # self.ax_toggle.spines['top'].set_visible(False)
        # self.ax_toggle.spines['right'].set_visible(False)
        # self.ax_toggle.spines['left'].set_visible(False)
        # self.ax_toggle.spines['bottom'].set_visible(False)

        # # Function to update the overlay toggle
        # def toggle_overlay(label):
        #     nonlocal overlay_on
        #     overlay_on = not overlay_on
        #     update(None)  # Re-render the figure with the updated overlay status

        # self.toggle_button.on_clicked(toggle_overlay)
        #
        # # Function to update the scan type when radio button is clicked
        # def update_scan_type(label):
        #     nonlocal current_scan
        #     current_scan = label.lower()  # Update the scan to the selected one
        #     update(None)  # Re-render the figure with the new scan type
        #
        # self.radio_button.on_clicked(update_scan_type)



        # # Link the update function to the sliders
        # self.slice_slider.on_changed(update)
        # self.time_slider.on_changed(update)
        # Set the background color of the figure and axes
        self.ax_sagittal.set_facecolor('black')
        self.ax_coronal.set_facecolor('black')
        self.ax_axial.set_facecolor('black')
        self.fig = fig
        return fig

        # Update function for the sliders and toggle
    def update(self, slice_idx, time_step, overlay, cur_scan):
        # slice_idx = int(self.slice_slider.val)
        # time_step = int(self.time_slider.val)

        def create_brain_mask(mri_image):
            brain_mask = mri_image > 0
            return brain_mask

        self.brain_mask_sagittal = create_brain_mask(self.mri_data['flair'][slice_idx, :, :])
        self.brain_mask_coronal = create_brain_mask(self.mri_data['flair'][:, slice_idx, :])
        self.brain_mask_axial = create_brain_mask(self.mri_data['flair'][:, :, slice_idx])

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

        # Update the images with the new slice and tumor mask
        self.scan_img_sagittal.set_data(self.scan_rgb_sagittal)
        self.scan_img_coronal.set_data(self.scan_rgb_coronal)
        self.scan_img_axial.set_data(self.scan_rgb_axial)

        self.fig.canvas.draw_idle()

    # Step 4: Interactive Visualization with Slice, Time Sliders, and Overlay Toggle
    def interactive_growth_visualization(self, mri_data):
        sagittal_slice_idx = mri_data['flair'].shape[0] // 2  # Start at the middle slice along the z-axis (sagittal)
        coronal_slice_idx = mri_data['flair'].shape[1] // 2  # Start at the middle slice along the x-axis (coronal)
        axial_slice_idx = mri_data['flair'].shape[2] // 2
        # Get the initial tumor mask for sagittal and coronal slices
        initial_tumor_mask_sagittal = mri_data['glistrboost'][sagittal_slice_idx, :, :] > 0
        initial_tumor_mask_coronal = mri_data['glistrboost'][:, coronal_slice_idx, :] > 0
        initial_tumor_mask_axial = mri_data['glistrboost'][:, :, axial_slice_idx] > 0
        # Resize the tumor masks to match the slice dimensions
        tumor_mask_resized_sagittal = self.resize_mask_to_slice(initial_tumor_mask_sagittal, mri_data['flair'].shape[1:3])
        tumor_mask_resized_coronal = self.resize_mask_to_slice(initial_tumor_mask_coronal, mri_data['flair'].shape[1:3])
        tumor_mask_resized_axial = self.resize_mask_to_slice(initial_tumor_mask_axial, mri_data['flair'].shape[:2])
        # Set up the figure and axis
        fig, (ax_sagittal, ax_coronal, ax_axial) = plt.subplots(1, 3, figsize=(14, 7))
        plt.subplots_adjust(left=0.25, bottom=0.35)

        # Initial scan setup for both figures
        current_scan = 'flair'
        scan_slice_sagittal = mri_data[current_scan][sagittal_slice_idx, :, :].T  # sagittal slice (z-axis)
        scan_slice_coronal = mri_data[current_scan][:, coronal_slice_idx, :].T  # coronal slice (x-axis)
        scan_slice_axial = mri_data[current_scan][:, :, axial_slice_idx].T

        scan_rgb_sagittal = np.repeat(scan_slice_sagittal[:, :, np.newaxis], 3, axis=2)
        scan_rgb_coronal = np.repeat(scan_slice_coronal[:, :, np.newaxis], 3, axis=2)
        scan_rgb_axial = np.repeat(scan_slice_axial[:, :, np.newaxis], 3, axis=2)

        # Normalize scan_rgb to ensure values are within [0, 1]
        scan_rgb_sagittal = np.clip(scan_rgb_sagittal / np.max(scan_rgb_sagittal), 0, 1)
        scan_rgb_coronal = np.clip(scan_rgb_coronal / np.max(scan_rgb_coronal), 0, 1)
        scan_rgb_axial = np.clip(scan_rgb_axial / np.max(scan_rgb_axial), 0, 1)

        # Display the initial tumor mask (default to FLAIR)
        scan_img_sagittal = ax_sagittal.imshow(scan_rgb_sagittal, origin='lower')
        scan_img_coronal = ax_coronal.imshow(scan_rgb_coronal, origin='lower')
        scan_img_axial = ax_axial.imshow(scan_rgb_axial, origin='lower')
        # Apply initial red overlay for the tumor region
        tumor_overlay_sagittal = tumor_mask_resized_sagittal.T
        tumor_overlay_coronal = tumor_mask_resized_coronal.T
        tumor_overlay_axial= tumor_mask_resized_axial.T
        overlay_on = True  # Control variable for overlay toggle

        # Apply tumor overlays
        scan_rgb_sagittal[tumor_overlay_sagittal, 0] = 1
        scan_rgb_sagittal[tumor_overlay_sagittal, 1] = 0
        scan_rgb_sagittal[tumor_overlay_sagittal, 2] = 0

        scan_rgb_coronal[tumor_overlay_coronal, 0] = 1
        scan_rgb_coronal[tumor_overlay_coronal, 1] = 0
        scan_rgb_coronal[tumor_overlay_coronal, 2] = 0

        scan_rgb_axial[tumor_overlay_axial, 0] = 1
        scan_rgb_axial[tumor_overlay_axial, 1] = 0
        scan_rgb_axial[tumor_overlay_axial, 2] = 0

        scan_img_sagittal.set_data(scan_rgb_sagittal)
        scan_img_coronal.set_data(scan_rgb_coronal)
        scan_img_axial.set_data(scan_rgb_axial)

        ax_slice_slider = plt.axes([0.25, 0.2, 0.65, 0.03])
        min_slices = self.get_max_slice_value(mri_data, current_scan)
        slice_slider = Slider(ax_slice_slider, 'Slice Index', 0, min_slices - 1, valinit=sagittal_slice_idx, valstep=1)

        # Slider for controlling time step
        ax_time_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        time_slider = Slider(ax_time_slider, 'Time Step', 0, EquationConstant.NUM_STEPS, valinit=0, valstep=1)

        def update_time_step(val):
            calculated_time = calculate_time_in_days(val)
            time_slider.valtext.set_text(f"{calculated_time:.2f} days")

        def calculate_time_in_days(step):
            time_step = (EquationConstant.SPATIAL_RESOLUTION ** 2) / (2 * 3 * self.diffusion_rate)
            return step * time_step

        time_slider.on_changed(update_time_step)

        # Checkbox for toggling red overlay
        ax_toggle = plt.axes([0.05, 0.5, 0.15, 0.15])
        toggle_button = CheckButtons(ax_toggle, ['Toggle Overlay'], [overlay_on])

        # RadioButtons for selecting scan type (FLAIR, T1, T1_Gd, etc.)
        ax_radio = plt.axes([0.05, 0.8, 0.15, 0.15])
        radio_button = RadioButtons(ax_radio, ['FLAIR', 'T1', 'T1 GD', 'T2'])

        # Hide the border around the checkbox
        for label in toggle_button.labels:
            label.set_fontsize(10)
            label.set_color('black')

        ax_toggle.spines['top'].set_visible(False)
        ax_toggle.spines['right'].set_visible(False)
        ax_toggle.spines['left'].set_visible(False)
        ax_toggle.spines['bottom'].set_visible(False)

        # Function to update the overlay toggle
        def toggle_overlay(label):
            nonlocal overlay_on
            overlay_on = not overlay_on
            update(None)  # Re-render the figure with the updated overlay status
        toggle_button.on_clicked(toggle_overlay)

        # Function to update the scan type when radio button is clicked
        def update_scan_type(label):
            nonlocal current_scan
            current_scan = label.lower()  # Update the scan to the selected one
            update(None)  # Re-render the figure with the new scan type
        radio_button.on_clicked(update_scan_type)

        # Update function for the sliders and toggle
        def update(val):
            slice_idx = int(slice_slider.val)
            time_step = int(time_slider.val)

            def create_brain_mask(mri_image):
                brain_mask = mri_image > 0
                return brain_mask

            brain_mask_sagittal = create_brain_mask(mri_data['flair'][slice_idx, :, :])
            brain_mask_coronal = create_brain_mask(mri_data['flair'][:, slice_idx, :])
            brain_mask_axial = create_brain_mask(mri_data['flair'][:, :, slice_idx])

            # Resize the brain mask to the shape of the current slice
            brain_mask_resized_sagittal = self.resize_mask_to_slice(brain_mask_sagittal, mri_data[current_scan].shape[1:])
            brain_mask_resized_coronal = self.resize_mask_to_slice(brain_mask_coronal, mri_data[current_scan].shape[1:])
            brain_mask_resized_axial = self.resize_mask_to_slice(brain_mask_axial, mri_data[current_scan].shape[:2])

            # Update the selected scan slice for both sagittal and coronal
            scan_slice_sagittal = mri_data[current_scan][slice_idx, :, :].T
            scan_slice_coronal = mri_data[current_scan][:, slice_idx, :].T
            scan_slice_axial = mri_data[current_scan][:, :, slice_idx].T
            scan_rgb_sagittal = np.repeat(scan_slice_sagittal[:, :, np.newaxis], 3, axis=2)
            scan_rgb_coronal = np.repeat(scan_slice_coronal[:, :, np.newaxis], 3, axis=2)
            scan_rgb_axial = np.repeat(scan_slice_axial[:, :, np.newaxis], 3, axis=2)
            # Normalize scan_rgb to ensure values are within [0, 1]
            scan_rgb_sagittal = np.clip(scan_rgb_sagittal / np.max(scan_rgb_sagittal), 0, 1)
            scan_rgb_coronal = np.clip(scan_rgb_coronal / np.max(scan_rgb_coronal), 0, 1)
            scan_rgb_axial = np.clip(scan_rgb_axial / np.max(scan_rgb_axial), 0, 1)
            # Resize the initial mask to match the new slice and simulate growth
            tumor_mask_resized_sagittal = self.resize_mask_to_slice(mri_data['glistrboost'][slice_idx, :, :] > 0, mri_data[current_scan].shape[1:])
            tumor_mask_resized_coronal = self.resize_mask_to_slice(mri_data['glistrboost'][:, slice_idx, :] > 0, mri_data[current_scan].shape[1:])
            tumor_mask_resized_axial = self.resize_mask_to_slice(mri_data['glistrboost'][:, :, slice_idx] > 0, mri_data[current_scan].shape[:2])
            grown_tumor_mask_sagittal = self.simulate_growth(tumor_mask_resized_sagittal, diffusion_rate=self.diffusion_rate, reaction_rate=self.reaction_rate, time_steps=time_step, brain_mask=brain_mask_sagittal)
            grown_tumor_mask_coronal = self.simulate_growth(tumor_mask_resized_coronal, diffusion_rate=self.diffusion_rate, reaction_rate=self.reaction_rate, time_steps=time_step, brain_mask=brain_mask_coronal)
            grown_tumor_mask_axial = self.simulate_growth(tumor_mask_resized_axial, diffusion_rate=self.diffusion_rate, reaction_rate=self.reaction_rate, time_steps=time_step, brain_mask=brain_mask_axial)

            # Apply tumor overlays
            if overlay_on:
                scan_rgb_sagittal[grown_tumor_mask_sagittal.T, 0] = 1
                scan_rgb_sagittal[grown_tumor_mask_sagittal.T, 1] = 0
                scan_rgb_sagittal[grown_tumor_mask_sagittal.T, 2] = 0

                scan_rgb_coronal[grown_tumor_mask_coronal.T, 0] = 1
                scan_rgb_coronal[grown_tumor_mask_coronal.T, 1] = 0
                scan_rgb_coronal[grown_tumor_mask_coronal.T, 2] = 0

                scan_rgb_axial[grown_tumor_mask_axial.T, 0] = 1
                scan_rgb_axial[grown_tumor_mask_axial.T, 1] = 0
                scan_rgb_axial[grown_tumor_mask_axial.T, 2] = 0

            # Update the images with the new slice and tumor mask
            scan_img_sagittal.set_data(scan_rgb_sagittal)
            scan_img_coronal.set_data(scan_rgb_coronal)
            scan_img_axial.set_data(scan_rgb_axial)

            fig.canvas.draw_idle()

        # Link the update function to the sliders
        slice_slider.on_changed(update)
        time_slider.on_changed(update)
        # Set the background color of the figure and axes
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

    def start_equation(self, cur_scan):

        self.mri_data = self.load_mri_data(self.file_paths)  # Load the MRI data

        max_slices = self.get_max_slice_value(self.mri_data, cur_scan) - 1

        # Initialize the interactive visualization
        testFig = self.interactive_growth_visualization_2(self.mri_data, cur_scan)
        cur_slice_index = self.sagittal_slice_idx
        return testFig, cur_slice_index, max_slices

    def update_file_paths(self, path_key, path_value):
        self.file_paths[path_key] = path_value

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
    obj.interactive_growth_visualization(mri_data)
