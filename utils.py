# Utility functions

import numpy as np
from typing import Union, Tuple
from data_BEN import BENDataModule
from data_EuroSAT import EuroSATDataModule


def compute_channel_statistics_rs(
    data_module: Union[BENDataModule, EuroSATDataModule],
    percentile: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, std and percentile for each channel in BEN or EuroSAT.
    Computes for training dataset only to prevent leakage.

    Parameters:
        data_module: Union[BENDataModule, EuroSATDataModule]
        percentile: float
            Percentile to compute (default: 99.0).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of (means, stds, percentile_values) for each channel.
    """
    # Process all training batches
    channel_values = None
    for batch in data_module.train_dataloader():
        images = batch[0].cpu().numpy()
        num_channels = images.shape[1]

        # Initialize channel_values if not done yet
        if channel_values is None:
            channel_values = [[] for _ in range(num_channels)]

        # Collect values for each channel
        for c in range(num_channels):
            channel_values[c].extend(images[:, c].flatten())

    # Compute statistics for each channel
    means = np.array([np.mean(channel) for channel in channel_values])
    stds = np.array([np.std(channel) for channel in channel_values])
    percentile_values = np.array([np.percentile(channel, percentile) for channel in channel_values])
    
    # Check if output dimensions are correct
    assert len(means) == num_channels, f"Expected {num_channels} channels, got {len(means)}"
    assert len(stds) == num_channels, f"Expected {num_channels} channels, got {len(stds)}"
    assert len(percentile_values) == num_channels, f"Expected {num_channels} channels, got {len(percentile_values)}"

    return means, stds, percentile_values