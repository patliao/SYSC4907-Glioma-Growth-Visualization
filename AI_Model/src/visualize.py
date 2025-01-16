import numpy as np
import matplotlib.pyplot as plt

def visualize_sample(data_dir, subject_id):
    # Load FLAIR image and segmentation mask
    flair_path = f"{data_dir}/{subject_id}/flair.npy"
    seg_path = f"{data_dir}/{subject_id}/seg.npy"

    flair = np.load(flair_path)
    seg = np.load(seg_path)

    # Visualize a slice
    slice_idx = flair.shape[2] // 2
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(flair[:, :, slice_idx], cmap="gray")
    plt.title("FLAIR")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(flair[:, :, slice_idx], cmap="gray")
    plt.imshow(seg[:, :, slice_idx], alpha=0.5, cmap="Reds")
    plt.title("Segmentation Overlay")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    visualize_sample("../data/processed", "BraTS2021_01476")
