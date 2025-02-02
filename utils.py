# Utility functions

import numpy as np
from typing import Union, Tuple, List
import torch
import os
from lightning.pytorch.callbacks import Callback

def compute_channel_statistics_rs(
    dataloader: torch.utils.data.DataLoader,
    percentile: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, std and percentile for each channel in BEN or EuroSAT.
    Computes for training dataset only to prevent leakage.
    Expects images in (B, C, H, W) format.
    
    First computes the percentile values, then uses these to normalize the data
    before computing mean and std, matching the normalization during training.

    Parameters:
        dataloader: torch.utils.data.DataLoader
            Dataloader for the training dataset
        percentile: float
            Percentile to compute (default: 99.0).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of (means, stds, percentile_values) for each channel.
    """
    # First pass: compute percentile values
    channel_values = None
    for batch in dataloader:
        images = batch[0].cpu().numpy()
        
        # Verify (B, C, H, W) format
        assert len(images.shape) == 4, f"Expected 4D tensor (B,C,H,W), got shape {images.shape}"
        batch_size, num_channels, height, width = images.shape
        
        # For remote sensing data, number of channels should be smaller than height/width
        assert num_channels < height and num_channels < width, \
            f"Expected channels ({num_channels}) to be smaller than height ({height}) and width ({width})"

        # Initialize channel_values if not done yet
        if channel_values is None:
            channel_values = [[] for _ in range(num_channels)]

        # Collect values for each channel
        for c in range(num_channels):
            channel_values[c].extend(images[:, c].flatten())

    # Compute percentile values
    percentile_values = np.array([np.percentile(channel, percentile) for channel in channel_values])
    
    # Second pass: compute mean and std on normalized data
    normalized_channel_values = [[] for _ in range(num_channels)]
    
    for batch in dataloader:
        images = batch[0].cpu().numpy()
        
        # Normalize each channel using its percentile value
        for c in range(num_channels):
            # Avoid division by zero
            denominator = percentile_values[c] if percentile_values[c] > 0 else 1e-6
            normalized_values = np.clip(images[:, c].flatten() / denominator, 0, 1)
            normalized_channel_values[c].extend(normalized_values)

    # Compute mean and std on normalized values
    means = np.array([np.mean(channel) for channel in normalized_channel_values])
    stds = np.array([np.std(channel) for channel in normalized_channel_values])
    
    # Check if output dimensions are correct
    assert len(means) == num_channels, f"Expected {num_channels} channels, got {len(means)}"
    assert len(stds) == num_channels, f"Expected {num_channels} channels, got {len(stds)}"
    assert len(percentile_values) == num_channels, f"Expected {num_channels} channels, got {len(percentile_values)}"

    return means, stds, percentile_values


def compute_channel_statistics_rgb(
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for each channel in Caltech101.
    Computes for training dataset only to prevent leakage.
    Assumes RGB images in range [0, 255] in (B, C, H, W) format.

    Parameters:
        dataloader: torch.utils.data.DataLoader
            Dataloader for the training dataset

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of (means, stds) for each channel.
    """
    # Process all training batches
    channel_values = [[] for _ in range(3)]  # RGB has 3 channels
    
    for batch in dataloader:
        images = batch[0].cpu().numpy()
        
        # Verify (B, C, H, W) format with 3 channels
        assert len(images.shape) == 4, f"Expected 4D tensor (B,C,H,W), got shape {images.shape}"
        batch_size, num_channels, height, width = images.shape
        assert num_channels == 3, f"Expected 3 channels for RGB, got {num_channels}"
        
        # Collect values for each channel
        for c in range(3):
            channel_values[c].extend(images[:, c].flatten())

    # Compute statistics for each channel
    means = np.array([np.mean(channel) for channel in channel_values])
    stds = np.array([np.std(channel) for channel in channel_values])
    
    # Check if output dimensions are correct
    assert len(means) == 3, f"Expected 3 channels, got {len(means)}"
    assert len(stds) == 3, f"Expected 3 channels, got {len(stds)}"

    return means, stds

class FeatureExtractionCallback(Callback):
    """Extracts penultimate feature representations at epochs 5 and 10."""

    def __init__(self):
        super().__init__()
        self.feature_representations = []
        self.labels = []
        # Create both directories
        os.makedirs("./untracked-files/features", exist_ok=True)
        os.makedirs("./untracked-files/features/extracted", exist_ok=True)

    def on_fit_start(self, trainer, pl_module):
        """Register hook on global pooling layer on forward pass."""
        assert hasattr(pl_module.model, 'global_pool'), "Global pooling layer not found with model.global_pool"
        
        # Hook function to collect activations
        def hook_function(module, input, output):
            # Check if trainer is in relevant epoch AND in training phase
            if trainer.current_epoch + 1 in [5, 10] and pl_module.training:  # Only collect during training
                feature_representation = output.detach().cpu().half()  # Convert to float16
                assert feature_representation.dtype == torch.float16, "Feature representation is not float16"
                self.feature_representations.append(feature_representation.numpy())  # Accumulate features
        
        self.hook = pl_module.model.global_pool.register_forward_hook(hook_function)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Store labels for each batch (ensuring correct order)."""
        epoch = trainer.current_epoch + 1  # starts usually with 0 
        _, labels = batch 
        if epoch in [5, 10]:
            # Convert one-hot encoded labels to class indices
            label_indices = torch.argmax(labels, dim=1)
            self.labels.extend(label_indices.cpu().numpy())

    def on_train_epoch_end(self, trainer, pl_module):
        """Save feature representations and labels at epochs 5 and 10."""
        epoch = trainer.current_epoch + 1  # starts usually with 0 
        if epoch in [5, 10] and self.feature_representations:
            features = np.concatenate(self.feature_representations, axis=0)  # Concatenate all batch features
            labels = np.array(self.labels)
            
            # Verify dimensions match
            assert len(features) == len(labels), f"Mismatch between features ({len(features)}) and labels ({len(labels)})"
            
            # Save to features/extracted (for version control)
            np.save(f"./untracked-files/features/extracted/epoch_{epoch}_features.npy", features)
            np.save(f"./untracked-files/features/extracted/epoch_{epoch}_labels.npy", labels)
            
            print(f"Saved features and labels for epoch {epoch}")
            # Clear buffers after saving
            self.feature_representations.clear()
            self.labels.clear()

    def on_fit_end(self, trainer, pl_module):
        """Remove hook."""
        self.hook.remove()
