import nibabel as nib
import os
import numpy as np
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go



# Define the path to the data folder
data_folder = 'data'

# Load the NIfTI files for each week
def load_nii_file(file_path):
    print(f"Loading file: {file_path}")  # Debugging: show file path being loaded
    img = nib.load(file_path)
    return img.get_fdata()

# Resize image to target shape using zoom
def resize_image(image_data, target_shape):
    print(f"Resizing image from shape {image_data.shape} to {target_shape}")  # Debugging: track resizing
    factors = np.array(target_shape) / np.array(image_data.shape)
    resized_image = zoom(image_data, factors, order=1)
    print(f"Resized image shape: {resized_image.shape}")  # Debugging: show new shape
    return resized_image

# Normalize image to [0, 1] range
def normalize_image(image_data):
    print(f"Normalizing image with min {np.min(image_data)} and max {np.max(image_data)}")  # Debugging: track normalization
    normalized_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    print(f"Normalized image min {np.min(normalized_image)} and max {np.max(normalized_image)}")  # Debugging: check range
    return normalized_image

# Load the images and masks for each week
def load_week_data(week_folder, target_shape=None):
    image_files = ['CT1.nii']
    mask_files = ['ct1_seg_mask.nii']
    
    images = {}
    masks = {}
    
    # Load images
    for img_file in image_files:
        img_path = os.path.join(data_folder, week_folder, img_file)
        img_data = load_nii_file(img_path)
        
        # Resize and normalize images
        if target_shape:
            img_data = resize_image(img_data, target_shape)
        img_data = normalize_image(img_data)
        
        images[img_file] = img_data
        print(f"Loaded and processed image {img_file} with shape {img_data.shape}")  # Debugging: confirm processing

    # Load masks
    for mask_file in mask_files:
        mask_path = os.path.join(data_folder, week_folder, mask_file)
        mask_data = load_nii_file(mask_path)
        
        # Resize the mask to the target shape
        if target_shape:
            mask_data = resize_image(mask_data, target_shape)
        
        # Ensure mask is binary (0 or 1)
        mask_data = np.where(mask_data > 0, 1, 0)
        masks[mask_file] = mask_data
        print(f"Processed mask {mask_file} with shape {mask_data.shape}")  # Debugging: check mask

    return images, masks

# Load the data and resize to a consistent target shape (e.g., 256x256x64)
target_shape = (128, 128, 64)  # Keeping high resolution for better visualization
week1_data = load_week_data('week000-1', target_shape)
week2_data = load_week_data('week044',target_shape)

# Convert image and mask dictionaries into arrays
def prepare_data_for_model(week1_data, week2_data):
    images = []
    masks = []
    
    # Week 1 data (training)
    images.append(week1_data[0]['CT1.nii'])
    masks.append(week1_data[1]['ct1_seg_mask.nii'])
    
    # Week 2 data (validation)
    images.append(week2_data[0]['CT1.nii'])
    masks.append(week2_data[1]['ct1_seg_mask.nii'])
    
    # Convert to numpy arrays
    images = np.array(images)
    masks = np.array(masks)
    
    # Debugging: Check the shape of the data
    print(f"Images shape: {images.shape}, Masks shape: {masks.shape}")
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.5, random_state=42)
    
    return X_train, X_val, y_train, y_val

# Prepare the data
X_train, X_val, y_train, y_val = prepare_data_for_model(week1_data, week2_data)

# Debugging: Print shapes of prepared data
print(f"\nTraining images shape: {X_train.shape}, Training masks shape: {y_train.shape}")
print(f"Validation images shape: {X_val.shape}, Validation masks shape: {y_val.shape}")

# Build the U-Net model
def build_unet_3d(input_shape):
    inputs = layers.Input(input_shape)
    
    # Encoder path
    conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D((2, 2, 2))(conv1)
    
    # Bottleneck
    conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    
    # Decoder path
    upconv1 = layers.Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2), padding='same')(conv2)
    merge1 = layers.concatenate([upconv1, conv1], axis=-1)
    conv3 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge1)
    
    # Output layer
    output = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv3)
    
    model = models.Model(inputs, output)
    return model

input_shape = (128, 128, 64, 1)  # Matches the target shape for images
model = build_unet_3d(input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Debugging: Model summary
model.summary()

# Callbacks
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# Train the model (2 epochs for quick debugging)
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=1,  # Keeping batch size 1 for large images (adjust if needed)
    epochs=2,  # Reduced epochs for quick debugging
    callbacks=[checkpoint, early_stopping],
    verbose=1
)
print("Training completed.")

# Visualization: Using the same data and visualization function
images = [week1_data[0]['CT1.nii'], week2_data[0]['CT1.nii']]
masks = [week1_data[1]['ct1_seg_mask.nii'], week2_data[1]['ct1_seg_mask.nii']]

# Function for linear interpolation of the mask
def interpolate_masks(mask_week1, mask_week2, num_frames=10):
    # Create a list of interpolated masks
    interpolated_masks = []
    
    # Linearly interpolate between week1 and week2 masks
    for i in range(num_frames):
        alpha = i / (num_frames - 1)  # alpha ranges from 0 to 1
        interpolated_mask = (1 - alpha) * mask_week1 + alpha * mask_week2
        interpolated_masks.append(interpolated_mask)
    
    return interpolated_masks

# Function to visualize the animation of glioma growth
def visualize_growth_animation(images, masks, week_idx=0, time_labels=["Week 1", "Week 2"], num_frames=10):
    # Select the data for week 1 and week 2
    img_week1 = images[0]
    img_week2 = images[1]
    mask_week1 = masks[0]
    mask_week2 = masks[1]
    
    # Normalize images and masks for visualization
    max_value_img = max(np.max(img_week1), np.max(img_week2))
    max_value_mask = max(np.max(mask_week1), np.max(mask_week2))
    
    img_week1 = img_week1 / max_value_img
    img_week2 = img_week2 / max_value_img
    mask_week1 = mask_week1 / max_value_mask
    mask_week2 = mask_week2 / max_value_mask
    
    # Interpolate the masks between week 1 and week 2
    interpolated_masks = interpolate_masks(mask_week1, mask_week2, num_frames)
    
    # Create the figure for the animation
    fig = go.Figure()
    
    # Add the brain images (static)
    fig.add_trace(go.Volume(
        x=np.repeat(np.arange(img_week1.shape[0]), img_week1.shape[1] * img_week1.shape[2]),
        y=np.tile(np.repeat(np.arange(img_week1.shape[1]), img_week1.shape[2]), img_week1.shape[0]),
        z=np.tile(np.arange(img_week1.shape[2]), img_week1.shape[0] * img_week1.shape[1]),
        value=img_week1.flatten(),
        isomin=0.1,
        isomax=0.9,
        opacity=0.3,
        surface_count=10,
        colorscale='Viridis',
        name=f'{time_labels[0]} - Image'
    ))
    
    # Add the initial glioma mask for week 1
    fig.add_trace(go.Volume(
        x=np.repeat(np.arange(mask_week1.shape[0]), mask_week1.shape[1] * mask_week1.shape[2]),
        y=np.tile(np.repeat(np.arange(mask_week1.shape[1]), mask_week1.shape[2]), mask_week1.shape[0]),
        z=np.tile(np.arange(mask_week1.shape[2]), mask_week1.shape[0] * mask_week1.shape[1]),
        value=mask_week1.flatten(),
        isomin=0.1,
        isomax=0.9,
        opacity=0.8,
        surface_count=5,
        colorscale='reds',
        name=f'{time_labels[0]} - Glioma Mask'
    ))
    
    # Add the final glioma mask for week 2
    fig.add_trace(go.Volume(
        x=np.repeat(np.arange(mask_week2.shape[0]), mask_week2.shape[1] * mask_week2.shape[2]),
        y=np.tile(np.repeat(np.arange(mask_week2.shape[1]), mask_week2.shape[2]), mask_week2.shape[0]),
        z=np.tile(np.arange(mask_week2.shape[2]), mask_week2.shape[0] * mask_week2.shape[1]),
        value=mask_week2.flatten(),
        isomin=0.1,
        isomax=0.9,
        opacity=0.8,
        surface_count=5,
        colorscale='reds',
        name=f'{time_labels[1]} - Glioma Mask'
    ))
    
    # Create the animation frames for the interpolated masks
    frames = []
    for i, mask in enumerate(interpolated_masks):
        frames.append(go.Frame(
            data=[go.Volume(
                x=np.repeat(np.arange(mask_week1.shape[0]), mask_week1.shape[1] * mask_week1.shape[2]),
                y=np.tile(np.repeat(np.arange(mask_week1.shape[1]), mask_week1.shape[2]), mask_week1.shape[0]),
                z=np.tile(np.arange(mask_week1.shape[2]), mask_week1.shape[0] * mask_week1.shape[1]),
                value=mask.flatten(),
                isomin=0.1,
                isomax=0.9,
                opacity=0.8,
                surface_count=5,
                colorscale='reds',
                name=f"Interpolated Glioma Mask - Frame {i+1}"
            )],
            name=f"Frame {i+1}"
        ))

    # Add animation settings
    fig.update_layout(
        title="Glioma Growth Animation",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                         method="animate",
                         args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])]
        )]
    )
    
    # Add frames to the figure
    fig.frames = frames
    
    # Show the animation
    fig.show()

# Visualize the animation of glioma growth between week 1 and week 2
visualize_growth_animation(images, masks, week_idx=0)