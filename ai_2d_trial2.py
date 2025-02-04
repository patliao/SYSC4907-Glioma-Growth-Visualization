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

# Load the NIfTI files
def load_nii_file(file_path):
    print(f"Loading file: {file_path}")
    img = nib.load(file_path)
    return img.get_fdata()

# Resize image to target shape
def resize_image(image_data, target_shape):
    print(f"Resizing image from {image_data.shape} to {target_shape}")
    factors = np.array(target_shape) / np.array(image_data.shape)
    return zoom(image_data, factors, order=1)

# Normalize image
def normalize_image(image_data):
    return (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

# Extract a **2D slice** from the middle of the 3D volume
def extract_2d_slice(image_data, slice_index=32):
    return image_data[:, :, slice_index]

# Load 2D slices from NIfTI files
def load_week_data(week_folder, target_shape=(128, 128), slice_index=32):
    image_file = os.path.join(data_folder, week_folder, 'CT1.nii')
    mask_file = os.path.join(data_folder, week_folder, 'ct1_seg_mask.nii')

    # Load images
    img_data = extract_2d_slice(load_nii_file(image_file), slice_index)
    img_data = resize_image(img_data, target_shape)
    img_data = normalize_image(img_data)

    # Load masks
    mask_data = extract_2d_slice(load_nii_file(mask_file), slice_index)
    mask_data = resize_image(mask_data, target_shape)
    mask_data = np.where(mask_data > 0, 1, 0)  # Convert to binary mask

    return img_data, mask_data

# Load the data
target_shape = (128, 128)
week1_data = load_week_data('week000-1', target_shape)
week2_data = load_week_data('week044', target_shape)

# Convert data to numpy arrays
X = np.array([week1_data[0], week2_data[0]])[..., np.newaxis]  # Add channel dimension
y = np.array([week1_data[1], week2_data[1]])[..., np.newaxis]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

# Build the 2D U-Net model
def build_unet_2d(input_shape):
    inputs = layers.Input(input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    # Bottleneck
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)

    # Decoder
    upconv1 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv2)
    merge1 = layers.concatenate([upconv1, conv1], axis=-1)
    conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(merge1)

    # Output
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv3)

    model = models.Model(inputs, output)
    return model

# Create and compile the model
input_shape = (128, 128, 1)
model = build_unet_2d(input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
checkpoint = ModelCheckpoint('best_model_2d.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=1,
    epochs=2,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)
print("Training completed.")

# Visualization

# Function to interpolate masks
def interpolate_masks(mask_week1, mask_week2, num_frames=10):
    interpolated_masks = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        interpolated_mask = (1 - alpha) * mask_week1 + alpha * mask_week2
        interpolated_masks.append(interpolated_mask)
    return interpolated_masks

# Function to visualize glioma growth animation
def visualize_growth_animation(images, masks, num_frames=10):
    mask_week1, mask_week2 = masks
    img_week1, img_week2 = images

    interpolated_masks = interpolate_masks(mask_week1, mask_week2, num_frames)

    fig = go.Figure()

    for i, mask in enumerate(interpolated_masks):
        fig.add_trace(go.Heatmap(z=mask, colorscale='reds', showscale=False, name=f"Frame {i+1}"))

    fig.update_layout(
        title="Glioma Growth Animation",
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
            }]
        }]
    )

    fig.frames = [go.Frame(data=[go.Heatmap(z=mask, colorscale='reds', showscale=False)]) for mask in interpolated_masks]

    fig.show()

# Function to visualize toggle between Week 1 & Week 2
def visualize_growth_toggle(images, masks):
    mask_week1, mask_week2 = masks
    img_week1, img_week2 = images

    fig = go.Figure()

    fig.add_trace(go.Heatmap(z=mask_week1, colorscale='reds', showscale=False, name="Week 1", visible=True))
    fig.add_trace(go.Heatmap(z=mask_week2, colorscale='reds', showscale=False, name="Week 2", visible=False))

    fig.update_layout(
        title="Glioma Growth - Week Comparison",
        updatemenus=[{
            'buttons': [
                {"label": "Week 1", "method": "update", "args": [{"visible": [True, False]}]},
                {"label": "Week 2", "method": "update", "args": [{"visible": [False, True]}]}
            ],
            "direction": "down",
            "showactive": True,
        }]
    )

    fig.show()

# Show the animation
visualize_growth_animation([week1_data[0], week2_data[0]], [week1_data[1], week2_data[1]])

# Show the toggle comparison
visualize_growth_toggle([week1_data[0], week2_data[0]], [week1_data[1], week2_data[1]])
