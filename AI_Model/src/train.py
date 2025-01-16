import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import Glioma3DCNN
import numpy as np
import nibabel as nib


def determine_label(seg_file):
    """
    Determine label based on the segmentation file content.
    Example: Assign 0 for less tumor involvement, 1 for more involvement.
    """
    seg_data = nib.load(seg_file).get_fdata()
    unique_values = np.unique(seg_data)
    # Placeholder logic: adjust based on your actual data
    return 0 if len(unique_values) <= 2 else 1


def load_data(data_dir):
    """
    Load and preprocess data from the processed directory.
    Assign labels based on the segmentation files.
    """
    X, y = [], []
    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if os.path.isdir(subject_path):
            # Load FLAIR modality as input data
            flair_file = os.path.join(subject_path, "flair.npy")
            seg_file = os.path.join(subject_path, "seg.npy")
            if os.path.exists(flair_file) and os.path.exists(seg_file):
                X.append(np.load(flair_file))  # Use FLAIR as input
                label = determine_label(os.path.join(data_dir, subject, "seg.npy"))
                y.append(label)
            else:
                print(f"Missing files for subject {subject}, skipping...")

    # Convert lists to tensors
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} samples with labels: {y}")
    return torch.tensor(X).unsqueeze(1).float(), torch.tensor(y).long()


def train_model(data_dir, num_epochs=5, batch_size=4, learning_rate=0.001):
    """
    Train the 3D CNN model using preprocessed data.
    """
    # Load data
    X, y = load_data(data_dir)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = Glioma3DCNN(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Save the trained model
    os.makedirs("../models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../models/checkpoints/glioma_model.pth")
    print("Model training complete and saved.")


if __name__ == "__main__":
    train_model("../data/processed")
