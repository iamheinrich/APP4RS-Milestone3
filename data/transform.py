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
    Optionally we canspecify a dtype (float32, uint16, etc.). Default is float32.
    """
    def __init__(self, dtype=torch.float32):
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
    precomputed 99th-percentile. Then clip and convert
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
# 3) Albumentations Wrappers
# -------------------------------
class AlbumentationsWrapper:
    """Base wrapper to make Albumentations transforms compatible with torchvision.transforms.Compose"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Apply Albumentations transform
        result = self.transform(image=x)['image']
        return result

class SharpenWrapper(AlbumentationsWrapper):
    # Works with float32 and uint8
    # Works with any number of channels
    def __init__(self):
        super().__init__(A.Sharpen())

class RandomResizeCropWrapper(AlbumentationsWrapper):
    # Works with float32 and uint8
    # Should work with any number of channels
    def __init__(self):
        super().__init__(A.RandomResizedCrop(height=100, width=100)) #TODO outputdim == inputdim??

class CutOutWrapper(AlbumentationsWrapper):
    # Works with float32 and uint8
    # Seems to work with any number of channels 
    # Using float values for height and width to define fractions, which should work for uint8 and float32
    def __init__(self):
        super().__init__(A.CoarseDropout(hole_height_range=(0.05,0.05), hole_width_range=(0.05,0.05), fill='random'))

class BrightnessWrapper(AlbumentationsWrapper):
    # Works with float32 and uint8
    # Works with any number of channels
    def __init__(self):
        super().__init__(A.RandomBrightnessContrast(contrast_limit=(0,0))) # Set contrast to 0 to ensure we only change brightness

class ContrastWrapper(AlbumentationsWrapper):
    # Works with float32 and uint8
    # Works with any number of channels
    def __init__(self):
        super().__init__(A.RandomBrightnessContrast(brightness_limit=(0,0))) # Set brightness to 0 to ensure we only change contrast

class ToGrayWrapper(AlbumentationsWrapper):
    # Works with float32 and uint8
    # Works with 3 channels
    def __init__(self):
        super().__init__(A.ToGray(num_output_channels=3)) # Converts back to 3 channels for RGB

class HorizontalVerticalFlipWrapper(AlbumentationsWrapper):
    #       datatypes ?
    # Works with any number of channels
    def __init__(self):
        transform = T.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip()
        ])
        super().__init__(transform)

class Resize112Wrapper(AlbumentationsWrapper):
    # (H, W, C) -> (112, 112, C)
    def __init__(self):
        super().__init__(A.Resize(height=112, width=112))

# -------------------------------
# 4) Multi-Channel GrayScale
# -------------------------------
class MultiChannelGrayScale:
    """
    Convert multi-channel (H, W, C) to grayscale by averaging across channels.
    For remote sensing data, maintains the same number of output channels.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(x)}")
        if x.ndim != 3:
            raise ValueError(f"Expected (H, W, C), got shape {x.shape}")
        
        # Calculate grayscale values by averaging across channels
        gray = x.mean(axis=2, keepdims=True)  # (H, W, 1)
        
        # Repeat the grayscale values to match the input number of channels
        num_channels = x.shape[2]
        return np.repeat(gray, num_channels, axis=2)  # (H, W, C)


# -------------------------------
# 5) Pipeline Builders
# -------------------------------
def build_pil_transform_pipeline(
    mean=None,
    std=None,
    apply_random_resize_crop=False,
    apply_cutout=False,
    apply_brightness=False,
    apply_contrast=False,
    apply_grayscale=False,
    apply_sharpen=False,
):
    """
    Build a transform pipeline for regular PIL images (RGB).

    Parameters:
        mean, std: list or tuple
            Mean and std for final normalization (one entry per channel).
        apply_random_resize_crop: bool
            Whether to apply random resize
        apply_cutout: bool
            Whether to apply random erasing
        apply_brightness: bool
            Whether to apply random brightness adjustment
        apply_contrast: bool
            Whether to apply random contrast adjustment
        apply_grayscale: bool
            Whether to convert to grayscale with 3 output channels
        apply_sharpen: bool
            Whether to apply sharpening
    Returns:
        torchvision.transforms.Compose that:
          1) Converts input to NumPy (H, W, C)
          2) Applies the specified augmentations if any
          3) Converts to Tensor (C, H, W) in float32
          4) Applies PyTorch Normalize(mean, std)
    """
    if mean is None or std is None:
        raise ValueError("Must provide mean and std for final normalization.")
    
    transform_list = []

    # Convert to NumPy
    transform_list.append(ToNumpy())

    # Optional augmentations
    if apply_random_resize_crop:
        transform_list.append(RandomResizeCropWrapper())
    if apply_cutout:
        transform_list.append(CutOutWrapper())
    if apply_brightness:
        transform_list.append(BrightnessWrapper())
    if apply_contrast:
        transform_list.append(ContrastWrapper())
    if apply_grayscale:
        transform_list.append(ToGrayWrapper())
    if apply_sharpen:
        transform_list.append(SharpenWrapper())

    # Convert back to Tensor
    transform_list.append(ToTensorCHW(dtype=torch.float32))

    # Final normalization - always applied if mean/std provided
    if mean is not None and std is not None:
        transform_list.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transform_list)


def build_rs_transform_pipeline(
    percentile_values,
    mean=None,
    std=None,
    apply_random_resize_crop=False,
    apply_cutout=False,
    apply_brightness=False,
    apply_contrast=False,
    apply_grayscale=False,
    apply_sharpen=False,
    apply_flip=False,
    apply_resize112=False
):
    """
    Build a transform pipeline for remote sensing images (multi-channel uint16).

    Parameters:
        percentile_values: np.ndarray
            The 99th-percentile values for each channel, used for normalization.
        mean, std: list or tuple
            Mean and std for final normalization (one entry per channel).
        apply_random_resize_crop: bool
            Whether to apply random resize
        apply_cutout: bool
            Whether to apply random erasing
        apply_brightness: bool
            Whether to apply random brightness adjustment
        apply_contrast: bool
            Whether to apply random contrast adjustment
        apply_grayscale: bool
            Whether to convert to grayscale while maintaining the number of channels
        apply_sharpen: bool
            Whether to apply sharpening
        apply_flip: bool
            Whether to apply horizontal and vertical flip    
        apply_resize112: bool
            Whether to resize to 112x112 as ViT-tiny input
    Returns:
        torchvision.transforms.Compose that:
          1) Converts input to NumPy (H, W, C)
          2) Normalizes uint16 values using percentile_values
          3) Applies the specified augmentations
          4) Converts to Tensor (C, H, W) in float32
          5) Applies PyTorch Normalize(mean, std)
    """
    if percentile_values is None or mean is None or std is None:
        raise ValueError("Must provide percentile_values, mean, and std for remote sensing data.")

    transform_list = []

    # Convert to NumPy and normalize uint16
    transform_list.append(ToNumpy())
    transform_list.append(Uint16NormalizeNumpy(percentile_values))

    # Optional augmentations
    if apply_random_resize_crop:
        transform_list.append(RandomResizeCropWrapper())
    if apply_cutout:
        transform_list.append(CutOutWrapper())
    if apply_brightness:
        transform_list.append(BrightnessWrapper())
    if apply_contrast:
        transform_list.append(ContrastWrapper())
    if apply_grayscale:
        transform_list.append(MultiChannelGrayScale())
    if apply_sharpen:
        transform_list.append(SharpenWrapper())
    if apply_flip:
        transform_list.append(HorizontalVerticalFlipWrapper())
    if apply_resize112:
        transform_list.append(Resize112Wrapper())
    # Convert back to Tensor
    transform_list.append(ToTensorCHW(dtype=torch.float32))

    # Final normalization - always applied if mean/std provided
    if mean is not None and std is not None:
        transform_list.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transform_list)


# -------------------------------
# 6) Convenience Wrappers
# -------------------------------
def get_caltech_transform(
    mean,
    std,
    apply_random_resize_crop=False,
    apply_cutout=False,
    apply_brightness=False,
    apply_contrast=False,
    apply_grayscale=False,
    apply_sharpen=False,
):
    """
    For standard 3-channel PIL images like Caltech101.
    Mean and std have to be scaled according to image scale (0..255) or (0..1).
    """
    return build_pil_transform_pipeline(
        mean=mean,
        std=std,
        apply_random_resize_crop=apply_random_resize_crop,
        apply_cutout=apply_cutout,
        apply_brightness=apply_brightness,
        apply_contrast=apply_contrast,
        apply_grayscale=apply_grayscale,
        apply_sharpen=apply_sharpen,
    )


def get_remote_sensing_transform(
    percentile_values,
    mean,
    std,
    apply_random_resize_crop=False,
    apply_cutout=False,
    apply_brightness=False,
    apply_contrast=False,
    apply_grayscale=False,
    apply_sharpen=False,
    apply_flip=False,
    apply_resize112=False
):
    """
    For multispectral data like BEN or EuroSAT.
    Mean and std have to be scaled according to image scale (0..255) or (0..1).
    """
    return build_rs_transform_pipeline(
        percentile_values=percentile_values,
        mean=mean,
        std=std,
        apply_random_resize_crop=apply_random_resize_crop,
        apply_cutout=apply_cutout,
        apply_brightness=apply_brightness,
        apply_contrast=apply_contrast,
        apply_grayscale=apply_grayscale,
        apply_sharpen=apply_sharpen,
        apply_flip=apply_flip,
        apply_resize112=apply_resize112
    )
