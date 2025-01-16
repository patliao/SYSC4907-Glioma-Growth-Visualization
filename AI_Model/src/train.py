import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import Glioma3DCNN
import numpy as np

def load_data(data_dir):
    import nibabel as nib
    X, y = [], []
    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if os.path.isdir(subject_path):
            X.append(np.load(os.path.join(subject_path, "flair.npy")))  # Use FLAIR as input
            # Determine label from segmentation file
            seg_file = os.path.join(subject_path, f"{subject}_seg.nii.gz")
            if os.path.exists(seg_file):
                label = determine_label(seg_file)  # Define logic in determine_label
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"Labels: {y}")  # Debug labels
    return torch.tensor(X).unsqueeze(1).float(), torch.tensor(y).long()


def train_model(data_dir):
    X, y = load_data(data_dir)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = Glioma3DCNN(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # Adjust epoch count as needed
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            print(f"Outputs: {outputs}")  # Debug model output
            print(f"Labels: {labels}")    # Debug labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "../models/checkpoints/glioma_model.pth")
    print("Model training complete and saved.")


if __name__ == "__main__":
    train_model("../data/processed")
