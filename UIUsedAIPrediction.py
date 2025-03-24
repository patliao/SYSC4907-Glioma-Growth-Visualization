
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Encoder with fewer filters
        self.encoder1 = self.conv_block(in_channels, 16)  # Reduced filters
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)

        # Decoder with fewer filters
        self.decoder1 = self.upconv_block(128, 64)
        self.decoder2 = self.upconv_block(64, 32)
        self.decoder3 = self.upconv_block(32, 16)

        # Final layer
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool3d(e1, kernel_size=2, stride=2))
        e3 = self.encoder3(F.max_pool3d(e2, kernel_size=2, stride=2))
        e4 = self.encoder4(F.max_pool3d(e3, kernel_size=2, stride=2))

        # Decoder
        d1 = self.decoder1(e4)
        d2 = self.decoder2(d1 + e3)
        d3 = self.decoder3(d2 + e2)

        # Final layer
        out = self.final(d3 + e1)
        return torch.sigmoid(out)  # Sigmoid for binary segmentation

class UIUsedAIPrediction:
    _instance = None

    def __init__(self):
        self.prediction_result = None
        self.t1_flair_path = None
        self.t2_flair_path = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = UIUsedAIPrediction()
        return cls._instance

    # Function to load a .nii file
    def load_nii_file(self, file_path):
        img = nib.load(file_path)
        data = img.get_fdata()  # Get the image data as a numpy array
        return data

    # Normalize FLAIR images to [0, 1]
    def normalize_image(self, image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))

    # Normalize segmentation masks to [0, 1]
    def normalize_segmentation(self, segmentation):
        return (segmentation > 0).astype(np.float32)  # Convert to binary mask

    # Pad the tensor to make its dimensions divisible by 16
    def pad_to_divisible(self, tensor, divisible_by=16):
        depth, height, width = tensor.shape[-3:]
        pad_depth = (divisible_by - depth % divisible_by) % divisible_by
        pad_height = (divisible_by - height % divisible_by) % divisible_by
        pad_width = (divisible_by - width % divisible_by) % divisible_by

        # Pad the tensor (symmetric padding)
        padded_tensor = F.pad(tensor, (0, pad_width, 0, pad_height, 0, pad_depth))
        return padded_tensor

    def predict_using_ai(self, slice_index):
        # Copy past the code which are used for ai prediction in this script
        # This function should be called by ui

        # Load the saved model
        device = torch.device("cpu")  # Use "cpu" if no GPU is available
        model = UNet3D(in_channels=2, out_channels=1).to(device)
        model.load_state_dict(torch.load("glioma_unet3d_final.pth", map_location=device))
        model.eval()  # Set the model to evaluation mode

        time1_flair = self.load_nii_file(self.t1_flair_path)
        time2_flair = self.load_nii_file(self.t2_flair_path)

        time1_flair_normalized = self.normalize_image(time1_flair)
        time2_flair_normalized = self.normalize_image(time2_flair)

        # Stack time1 and time2 FLAIR images as input
        x = torch.tensor(np.stack([time1_flair_normalized, time2_flair_normalized], axis=0), dtype=torch.float32)
        x = self.pad_to_divisible(x)  # Pad the input tensor
        x = x.unsqueeze(0)  # Add batch dimension (batch size = 1)

        # Generate predictions
        with torch.no_grad():  # Disable gradient computation
            outputs = model(x)

        # Convert the output to a numpy array
        predicted_segmentation = outputs.squeeze().numpy()  # Remove batch and channel dimensions

        # Apply a threshold to the predicted segmentation
        threshold = 0.5
        predicted_segmentation_binary = (predicted_segmentation > threshold).astype(np.float32)

        # counter-clockwise rotate 90 degree to sync with equation model
        update_result =  np.rot90(predicted_segmentation_binary, 1)
        self.prediction_result = update_result.copy()

        # pred_slice = predicted_segmentation_binary[:, :, slice_idx]

        return self.get_slice_prediction(slice_index)

    def get_slice_prediction(self, slice_index):
        result = self.prediction_result[:, :, slice_index].copy()

        return result == 1

    def set_flair1(self, flair):
        self.t1_flair_path = flair

    def set_flair2(self, flair):
        self.t2_flair_path = flair