# implementing the CNN-based approach using transfer learning.
# integrates key steps for building and training a CNN using a pre-trained model like ResNet-50 to classify glioma MRI images

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess MRI Data
def load_and_preprocess_data(image_dir, target_size=(224, 224)):
    """
    Loads MRI images from a directory and preprocesses them for training.
    Args:
        image_dir (str): Path to the directory containing MRI images organized by label folders.
        target_size (tuple): Target size to resize images to (default: 224x224).
    Returns:
        images (numpy.ndarray): Preprocessed image data.
        labels (numpy.ndarray): Corresponding labels for the images.
    """
    images = []
    labels = []
    
    for label_folder in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label_folder)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size, color_mode='rgb')
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(label_folder)
    
    images = np.array(images) / 255.0  # Normalize pixel values to [0, 1]
    labels = np.array(labels)
    return images, labels

# Step 2: Prepare Data
image_dir = "path/to/processed/mri/images"  # Replace with the actual path
images, labels = load_and_preprocess_data(image_dir)

# Encode labels (e.g., 'low_grade', 'high_grade')
label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
encoded_labels = np.array([label_mapping[label] for label in labels])

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, encoded_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 3: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

# Step 4: Transfer Learning with ResNet-50
def build_transfer_learning_model():
    """
    Builds a CNN model using ResNet50 for transfer learning.
    Returns:
        model (tensorflow.keras.Model): Compiled CNN model.
    """
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers[:143]:  # Freeze first 143 layers
        layer.trainable = False
    
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(label_mapping), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_transfer_learning_model()

# Step 5: Train the Model
batch_size = 32
epochs = 10
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

history = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# Step 6: Evaluate and Save Model
test_predictions = model.predict(X_test)
test_predictions = np.argmax(test_predictions, axis=1)
print("Classification Report:\n", classification_report(y_test, test_predictions))
print("Accuracy:", accuracy_score(y_test, test_predictions))

model.save("glioma_classification_model.h5")
print("Model saved as glioma_classification_model.h5")

# Step 7: Visualize Training Results
def plot_training(history):
    """
    Plots training and validation accuracy/loss curves.
    Args:
        history (tensorflow.keras.callbacks.History): Training history object.
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_training(history)
