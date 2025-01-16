import torch
from model import Glioma3DCNN
from train import load_data

def evaluate_model(data_dir, checkpoint_path):
    # Load data
    X, y = load_data(data_dir)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # Load model
    model = Glioma3DCNN(num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    evaluate_model("../data/processed", "../models/checkpoints/glioma_model.pth")
