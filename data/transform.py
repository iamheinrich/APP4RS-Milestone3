import torch
import numpy as np
from PIL import Image

import albumentations as A
import torchvision.transforms as T


# -------------------------------
# 1) Basic Conversions
# -------------------------------
class ToNumpy:
    """
    Convert a PIL Image or Torch Tensor into a NumPy array of shape (H, W, C).
    """
    def __call__(self, x):
        # Handle PIL Images
        if isinstance(x, Image.Image):
            # PIL images are typically already (H, W) for grayscale or (H, W, 3) for RGB
            # np.array(...) yields (H, W, C) for RGB
            arr = np.array(x)
            # If it's grayscale with shape (H, W), make it (H, W, 1)
            if arr.ndim == 2:  # e.g. single-channel
                arr = np.expand_dims(arr, axis=2)  # (H, W, 1)
            return arr

        # Handle Torch Tensors
        elif isinstance(x, torch.Tensor):
            assert x.ndim == 3, f"Expected 3 dimensions (C,H,W) got shape {x.shape}"
            # We assume that in PyTorch channels are the leading dimension, but for generalizability let's assume shape is either (C, H, W) or (H, W, C)
            # We want to end with (H, W, C) to match numpy's default that's expected by albumentations.
            # In the case of BEN and EuroSAT, channels are the leading dimension and H,W to be larger than C.
            assert x.shape[0] < x.shape[1] and x.shape[0] < x.shape[2], f"Expected channels to be the leading dimension, got shape {x.shape}"
            # Permute to (H, W, C)
            x = x.permute(1, 2, 0)

            return x.cpu().numpy()

        else:
            raise TypeError(f"Unsupported input type for ToNumpy: {type(x)}")


class ToTensorCHW:
    """
    Convert a NumPy array (H, W, C) back to a torch.Tensor (C, H, W).
    Optionally we canspecify a dtype (float32, uint16, etc.). Default is uint16.
    """
    def __init__(self, dtype=torch.uint16):
        self.dtype = dtype

    def __call__(self, x: np.ndarray):
        assert x.ndim == 3, f"Expected (H, W, C), got shape {x.shape}"

        # Permute to (C, H, W)
        tensor = torch.from_numpy(x).permute(2, 0, 1)
        return tensor.to(self.dtype)


# -------------------------------
# 2) DataBEN and EuroSAT Normalization
# -------------------------------
class Uint16NormalizeNumpy:
    """
    Convert raw uint16 in each channel to [0..1] by dividing by a
    precomputed 99th-percentile. Then clip.
    """
    def __init__(self, percentile_values: np.ndarray):
        """
        percentile_values: shape (C,) specifying the 99th-percentile
                          for each channel.
        """
        self.percentile_values = percentile_values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x shape => (H, W, C), probably dtype uint16 or float
        if x.ndim != 3:
            raise ValueError(f"Expected (H, W, C), got shape {x.shape}")
        assert x.dtype == np.uint16, f"Expected dtype uint16, got {x.dtype}"

        h, w, c = x.shape
        if len(self.percentile_values) != c:
            raise ValueError(
                f"Mismatch channels: x.shape[2] = {c}, "
                f"percentile_values has length {len(self.percentile_values)}"
                "Both must have same length."
            )

        # Convert to float for safe division
        x = x.astype(np.float32)
        for ci in range(c):
            denominator = self.percentile_values[ci]
            if denominator == 0:
                denominator = 1e-6  # avoid division by zero
            x[..., ci] = np.clip(x[..., ci] / denominator, 0, 1)  # clip to [0..1]

        return x


# -------------------------------
# 3) Albumentations Pipeline
# -------------------------------
def build_albumentations_compose(
    apply_sharpen=True,
    apply_contrast=True,
    sharpen_p=0.5, # Default value
    contrast_p=0.5, # Default value
    overall_p=1.0 # Applies the entire pipeline with this probability
):
    """
    Build an Albumentations Compose with optional Sharpen & Contrast.
    Each transform has p=1.0 inside the Compose, but we set an 'overall_p'
    for the entire pipeline if we want to skip them sometimes.
    """
    atrans = []

    if apply_sharpen:
        atrans.append(
            A.Sharpen(
                alpha=(0.2, 0.5), # Default values
                lightness=(0.5, 1.0), # Default values
                p=sharpen_p  # Probability of applying Sharpen individually
            )
        )

    if apply_contrast:
        atrans.append(
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2), # Default values
                contrast_limit=(-0.2, 0.2), # Default values
                p=contrast_p  # Probability of applying RBC individually
            )
        )

    return A.Compose(atrans, p=overall_p)


class AlbumentationsTransform:
    """
    A PyTorch wrapper that applies an Albumentations Compose.
    Expects a NumPy array (H, W, C); returns a NumPy array (H, W, C).
    """
    def __init__(self, albumentations_compose: A.Compose):
        self.albumentations_compose = albumentations_compose

    def __call__(self, x: np.ndarray) -> np.ndarray:
        augmented = self.albumentations_compose(image=x)
        return augmented['image']


# -------------------------------
# 4) Optional Multi-Channel GrayScale
# -------------------------------
class MultiChannelGrayScale:
    """
    Convert multi-channel (H, W, C) to single-channel (H, W, 1) by averaging across channels.
    """
    def __call__(self, x: np.ndarray):
        if x.ndim != 3:
            raise ValueError(f"Expected (H, W, C), got {x.shape}")
        return np.mean(x, axis=2, keepdims=True)  # => (H, W, 1)


# -------------------------------
# 5) Pipeline Builders
# -------------------------------
def build_pil_transform_pipeline(
    mean=None,
    std=None,
    apply_augmentations=False,
    apply_sharpen=True,
    apply_contrast=True,
    sharpen_p=0.5,
    contrast_p=0.5,
    overall_aug_p=1.0,
    apply_grayscale=False
):
    """
    Build a transform pipeline for regular PIL images (RGB).

    Parameters:
        mean, std: list or tuple
            Mean and std for final PyTorch Normalize (one entry per channel).
        apply_augmentations: bool
            Whether to include Albumentations transforms.
        apply_sharpen, apply_contrast: bool
            Which augmentations to include if apply_augmentations=True.
        sharpen_p, contrast_p: float
            Individual probabilities for each augmentation.
        overall_aug_p: float
            Probability to apply the Albumentations pipeline as a whole.
        apply_grayscale: bool
            If True => convert to single-channel grayscale.

    Returns:
        torchvision.transforms.Compose that:
          1) Converts input to NumPy (H, W, C)
          2) Optionally applies Albumentations
          3) Optionally converts to grayscale
          4) Converts to Tensor (C, H, W) in float32
          5) Applies PyTorch Normalize(mean, std) if provided
    """
    transform_list = []

    # Convert input to NumPy
    transform_list.append(ToNumpy())

    # Albumentations-based augmentations
    if apply_augmentations:
        alb_compose = build_albumentations_compose(
            apply_sharpen=apply_sharpen,
            apply_contrast=apply_contrast,
            sharpen_p=sharpen_p,
            contrast_p=contrast_p,
            overall_p=overall_aug_p
        )
        transform_list.append(AlbumentationsTransform(alb_compose))

    # Optional multi-channel grayscale
    if apply_grayscale:
        transform_list.append(MultiChannelGrayScale())

    # Convert to Tensor (float32)
    transform_list.append(ToTensorCHW(dtype=torch.float32))

    # Final normalization if mean/std provided
    if mean is not None and std is not None:
        transform_list.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transform_list)


def build_rs_transform_pipeline(
    percentile_values,
    mean=None,
    std=None,
    apply_augmentations=False,
    apply_sharpen=True,
    apply_contrast=True,
    sharpen_p=0.5,
    contrast_p=0.5,
    overall_aug_p=1.0,
    apply_grayscale=False
):
    """
    Build a transform pipeline for remote sensing images (multi-channel uint16).

    Parameters:
        percentile_values: np.ndarray
            The 99th-percentile values for each channel, used for normalization. Must be the same length as number of channels in input data.
        mean, std: list or tuple
            Mean and std for final PyTorch Normalize (one entry per channel).
        apply_augmentations: bool
            Whether to include Albumentations transforms.
        apply_sharpen, apply_contrast: bool
            Which augmentations to include if apply_augmentations=True.
        sharpen_p, contrast_p: float
            Individual probabilities for each augmentation.
        overall_aug_p: float
            Probability to apply the Albumentations pipeline as a whole.
        apply_grayscale: bool
            If True => convert to single-channel grayscale.

    Returns:
        torchvision.transforms.Compose that:
          1) Converts input to NumPy (H, W, C)
          2) Normalizes uint16 values using percentile_values
          3) Optionally applies Albumentations
          4) Optionally converts to grayscale
          5) Converts to Tensor (C, H, W) in float32
          6) Applies PyTorch Normalize(mean, std) if provided
    """
    if percentile_values is None:
        raise ValueError("Must provide percentile_values for remote sensing data.")

    transform_list = []

    # Convert input to NumPy
    transform_list.append(ToNumpy())

    # Handle uint16 normalization
    transform_list.append(Uint16NormalizeNumpy(percentile_values))

    # Albumentations-based augmentations
    if apply_augmentations:
        alb_compose = build_albumentations_compose(
            apply_sharpen=apply_sharpen,
            apply_contrast=apply_contrast,
            sharpen_p=sharpen_p,
            contrast_p=contrast_p,
            overall_p=overall_aug_p
        )
        transform_list.append(AlbumentationsTransform(alb_compose))

    # Optional multi-channel grayscale
    if apply_grayscale:
        transform_list.append(MultiChannelGrayScale())

    # Convert to Tensor (float32)
    transform_list.append(ToTensorCHW(dtype=torch.float32))

    # Final normalization if mean/std provided
    if mean is not None and std is not None:
        transform_list.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transform_list)


# -------------------------------
# 6) Convenience Wrappers
# -------------------------------
def get_caltech_transform(
    mean,
    std,
    apply_augmentations=False
):
    """
    For standard 3-channel PIL images like Caltech101.
    Mean and std have to be scaled according to image scale (0..255) or (0..1).
    """
    return build_pil_transform_pipeline(
        mean=mean,
        std=std,
        apply_augmentations=apply_augmentations
    )


def get_remote_sensing_transform(
    percentile_values,
    mean,
    std,
    apply_augmentations=False,
    apply_grayscale=False
):
    """
    For multispectral data like BEN or EuroSAT,
    Mean and std have to be scaled according to image scale (0..255) or (0..1).
    """
    return build_rs_transform_pipeline(
        percentile_values=percentile_values,
        mean=mean,
        std=std,
        apply_augmentations=apply_augmentations,
        apply_grayscale=apply_grayscale
    )
