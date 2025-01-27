# Utility functions

import numpy as np
from typing import Union, Tuple
from data_BEN import BENDataModule
from data_EuroSAT import EuroSATDataModule
from data.caltech101 import Caltech101DataModule


def compute_channel_statistics_rs(
    data_module: Union[BENDataModule, EuroSATDataModule],
    percentile: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, std and percentile for each channel in BEN or EuroSAT.
    Computes for training dataset only to prevent leakage.
    Expects images in (B, C, H, W) format.
    
    First computes the percentile values, then uses these to normalize the data
    before computing mean and std, matching the normalization during training.

    Parameters:
        data_module: Union[BENDataModule, EuroSATDataModule]
        percentile: float
            Percentile to compute (default: 99.0).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of (means, stds, percentile_values) for each channel.
    """
    # First pass: compute percentile values
    channel_values = None
    for batch in data_module.train_dataloader():
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
    
    for batch in data_module.train_dataloader():
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
    data_module: Caltech101DataModule,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for each channel in Caltech101.
    Computes for training dataset only to prevent leakage.
    Assumes RGB images in range [0, 255] in (B, C, H, W) format.

    Parameters:
        data_module: Caltech101DataModule

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of (means, stds) for each channel.
    """
    # Process all training batches
    channel_values = [[] for _ in range(3)]  # RGB has 3 channels
    
    for batch in data_module.train_dataloader():
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


