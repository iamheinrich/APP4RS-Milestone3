# partial functions
from functools import partial
from glob import glob
from hashlib import md5
from math import ceil
from typing import List, Literal, Optional

import lmdb
import pandas as pd
import rasterio
import torch
from lightning.pytorch import LightningDataModule
from safetensors.numpy import load as load_np_safetensor
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

from utils import compute_channel_statistics_rs
from data.transform import get_remote_sensing_transform, build_rs_transform_pipeline

def _hash(data):
    return md5(str(data).encode()).hexdigest()


EUROSAT_CLASSES = [
    "Forest",
    "AnnualCrop",
    "Highway",
    "HerbaceousVegetation",
    "Pasture",
    "Residential",
    "River",
    "Industrial",
    "PermanentCrop",
    "SeaLake"
]
EUROSAT_CLASSES.sort()

EUROSAT_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]


def _stringlist_to_tensor(cls: str) -> torch.Tensor:
    tensor = torch.zeros(len(EUROSAT_CLASSES))
    tensor[EUROSAT_CLASSES.index(cls)] = 1
    return tensor


class EuroSATIndexableLMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the EuroSAT dataset using an lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path
        self.transform = transform
        self.bandorder = bandorder
        self.env = None
        self.metadata = pd.read_parquet(metadata_parquet_path)
        if split is not None:
            self.metadata = self.metadata[self.metadata['split'] == split]
        self.keys = self.metadata['patch_name'].tolist()
        # sort keys to ensure reproducibility
        self.keys.sort()
        # Set default transform if none provided
        if transform is None:
            transform = build_rs_transform_pipeline(
                percentile_values=[10000] * len(bandorder),  # Example percentile values
                mean=[0.5] * len(bandorder),
                std=[0.5] * len(bandorder),
                apply_sharpness=True,
                apply_contrast=True,
                apply_grayscale=True
            )
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def _open_lmdb(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        self._open_lmdb()
        key = self.keys[idx]
        # get img data from lmdb
        with self.env.begin(write=False) as txn:
            patch = load_np_safetensor(txn.get(key.encode()))
        patch = torch.stack([torch.from_numpy(patch[k]) for k in self.bandorder]).float()
        if self.transform:
            patch = self.transform(patch)

        # get label from metadata
        label = self.metadata[self.metadata['patch_name'] == key]["class_name"]
        assert len(label) == 1
        label = label.values[0]
        label = _stringlist_to_tensor(label)
        return patch, label


class EuroSATIndexableTifDataset(Dataset):
    def __init__(self, base_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the EuroSAT dataset using tif files.

        :param base_path: path to the source EuroSAT dataset (root of the zip file)
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        self.tif_path = base_path
        self.transform = transform
        self.bandorder = bandorder
        self.keys = glob(self.tif_path + '/*/*.tif')
        if split is not None:
            self.keys = {
                cls: sorted([key for key in self.keys if cls in key],
                            key=lambda x: int(x.split('/')[-1].split('_')[1].split('.')[0]))
                for cls in EUROSAT_CLASSES
            }
            if split == 'train':
                # take first 70% of every class
                self.keys = {
                    cls: keys[:int(0.7 * len(keys))]
                    for cls, keys in self.keys.items()
                }
            elif split == "validation":
                # take next 15% of every class
                self.keys = {
                    cls: keys[int(0.7 * len(keys)):int(0.85 * len(keys))]
                    for cls, keys in self.keys.items()
                }
            elif split == "test":
                # take last 15% of every class
                self.keys = {
                    cls: keys[int(0.85 * len(keys)):]
                    for cls, keys in self.keys.items()
                }
            else:
                raise ValueError(f'Unknown split: {split}')
            # flatten keys
            self.keys = [key for cls in self.keys.values() for key in cls]
        # sort keys to ensure reproducibility
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: index of the item to get
        :return: (patch, label) tuple where patch is a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        key = self.keys[idx]
        data = rasterio.open(key).read()
        patch = {k: data[i] for i, k in enumerate(EUROSAT_BANDS)}
        patch = torch.stack([torch.from_numpy(patch[k]) for k in self.bandorder], dim=0).float()
        if self.transform:
            patch = self.transform(patch)

        # get label from name
        label = key.split('/')[-2]
        label = _stringlist_to_tensor(label)
        return patch, label


class EuroSATIterableLMDBDataset(IterableDataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None,
                 with_keys=False):
        """
        IterableDataset for the EuroSAT dataset using an lmdb file.

        :param lmdb_path: path to the lmdb file
        :param metadata_parquet_path: path to the metadata parquet file
        :param bandorder: order of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        self.lmdb_path = lmdb_path
        self.metadata_parquet_path = metadata_parquet_path
        self.transform = transform
        self.bandorder = bandorder
        self.with_keys = with_keys
        self.env = None
        self.env_iter = None
        self.metadata = pd.read_parquet(metadata_parquet_path)
        self.split = split
        if split is not None:
            self.metadata = self.metadata[self.metadata['split'] == split]
        self.keys = self.metadata['patch_name'].tolist()
        # sort keys to ensure reproducibility
        self.keys.sort()
        self.key_start = 0
        self.key_end = len(self.keys)
        self.initialized = None

    def _stack(self, patch):
        # interpolate each channel to 64x64
        patch = {
            k: torch.nn.functional.interpolate(
                torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            for k, v in patch.items()
        }
        # stack channels
        patch = torch.stack([patch[k] for k in self.bandorder], dim=0).float()
        return patch

    def _open_lmdb(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)
            self.env_iter = self.env.begin(write=False)

    def _process_init(self):
        if not self.initialized:
            self._open_lmdb()
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                self.key_start = 0
                self.key_end = len(self.keys)
                self.keys = set([k.encode() for k in self.keys])
            else:
                per_worker = ceil(len(self.keys) / worker_info.num_workers)
                worker_id = worker_info.id
                self.key_start = worker_id * per_worker
                self.key_end = min(self.key_start + per_worker, len(self.keys))
                self.keys = set([k.encode() for k in self.keys[self.key_start:self.key_end]])
            self.initialized = True

    def __len__(self):
        return self.key_end - self.key_start

    def __iter__(self):
        """
        Iterate over the dataset.

        :return: an iterator over the dataset, e.g. via `yield` where each item is a (patch, label) tuple where patch is
            a tensor of shape (C, H, W) and label is a tensor of shape (N,)
        """
        self._process_init()
        for key, data in self.env_iter.cursor():
            if key in self.keys:
                key = key.decode()
                patch = load_np_safetensor(data)
                patch = self._stack(patch)
                if self.transform:
                    patch = self.transform(patch)

                # get label from metadata
                label = self.metadata[self.metadata['patch_name'] == key]['class_name']
                assert len(label) == 1
                label = label.values[0]
                label = _stringlist_to_tensor(label)
                if self.with_keys:
                    yield patch, label, key
                else:
                    yield patch, label


class EuroSATDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            num_workers: int,
            bandorder: List,
            ds_type: Literal['iterable_lmdb', 'indexable_tif', 'indexable_lmdb'],
            base_path: Optional[str] = None,
            lmdb_path: Optional[str] = None,
            metadata_parquet_path: Optional[str] = None,
    ):
        """
        DataModule for the EuroSAT dataset.

        :param batch_size: batch size for the dataloaders
        :param num_workers: number of workers for the dataloaders
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param ds_type: type of dataset to use, one of 'iterable_lmdb', 'indexable_tif', 'indexable_lmdb'
        :param base_path: path to the source BigEarthNet dataset (root of the tar file), for tif dataset
        :param lmdb_path: path to the converted lmdb file, for lmdb dataset
        :param metadata_parquet_path: path to the metadata parquet file, for lmdb dataset
        """
        super().__init__()
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ds_type = ds_type
        self.bandorder = bandorder
        if ds_type == 'indexable_tif':
            assert base_path is not None, 'base_path must be provided for indexable_tif dataset'
            self.dataset = partial(EuroSATIndexableTifDataset,
                                   base_path=base_path,
                                   bandorder=bandorder)
        elif ds_type == 'iterable_lmdb':
            assert lmdb_path is not None, 'lmdb_path must be provided for iterable_lmdb dataset'
            assert metadata_parquet_path is not None, \
                'metadata_parquet_path must be provided for iterable_lmdb dataset'
            self.dataset = partial(EuroSATIterableLMDBDataset,
                                   lmdb_path=lmdb_path,
                                   metadata_parquet_path=metadata_parquet_path,
                                   bandorder=bandorder)
        elif ds_type == 'indexable_lmdb':
            assert lmdb_path is not None, 'lmdb_path must be provided for indexable_lmdb dataset'
            assert metadata_parquet_path is not None, \
                'metadata_parquet_path must be provided for indexable_lmdb dataset'
            self.dataset = partial(EuroSATIndexableLMDBDataset,
                                   lmdb_path=lmdb_path,
                                   metadata_parquet_path=metadata_parquet_path,
                                   bandorder=bandorder)
        else:
            raise ValueError(f'Unknown dataset type: {ds_type}')
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Adding mean, std and percentile values for training dataset
        self.mean = None
        self.std = None
        self.percentile = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # First create dataset without transforms to compute statistics so that we don't have to laod the dataset twice
            self.train_dataset = self.dataset(
                split='train',
                transform=None  # No transforms for statistics computation
            )
            
            temp_train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False  # No need to shuffle for statistics
            )
            
            # Compute statistics using only training data to prevent leakage
            self.mean, self.std, self.percentile = compute_channel_statistics_rs(
                temp_train_dataloader
            )
            
            # Training transform includes augmentations
            train_transform = get_remote_sensing_transform(
                percentile_values=self.percentile,
                mean=self.mean,
                std=self.std,
                # apply_brightness=True
                # Add augmentations as needed
            )
            
            # Validation and test transforms apply normalization only
            val_test_transform = get_remote_sensing_transform(
                percentile_values=self.percentile,
                mean=self.mean,
                std=self.std,
                # apply_brightness=True
                # Add augmentations as needed
            )
            
            # Update the transform attribute of the existing train_dataset
            self.train_dataset.transform = train_transform
            
            # Validation dataset without transforms
            self.val_dataset = self.dataset(
                split='validation',
                transform=val_test_transform
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset(
                split='test',
                transform=val_test_transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True if "indexable" in self.ds_type else None
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if "indexable" in self.ds_type else None
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if "indexable" in self.ds_type else None
        )


############################################ DON'T CHANGE CODE BELOW HERE ############################################

def main(
        lmdb_path: str,
        metadata_parquet_path: str,
        tif_base_path: str,
        bandorder: List,
        sample_indices: List[int],
        num_batches: int,
        seed: int,
        timing_samples: int,
):
    """
    Test the EuroSAT dataset classes

    :param lmdb_path: path to the converted lmdb file
    :param metadata_parquet_path: path to the metadata parquet file
    :param tif_base_path: path to the source BigEarthNet dataset (root of the tar file)
    :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
    :param sample_indices: indices of samples to check for correctness
    :param num_batches: number of batches to check in the dataloaders for correctness
    :param seed: seed for the dataloaders for reproducibility
    :param timing_samples: number of samples to check during timing
    :return: None
    """
    import time

    # check values of sample_indices
    for split in ['train', 'validation', 'test', None]:
        print(f"\nSplit: {split}")
        for DS in [EuroSATIndexableLMDBDataset, EuroSATIndexableTifDataset, EuroSATIterableLMDBDataset]:
            paths = {
                "base_path": tif_base_path,
            } if DS == EuroSATIndexableTifDataset else {
                "lmdb_path": lmdb_path,
                "metadata_parquet_path": metadata_parquet_path,
            }
            ds = DS(
                bandorder=bandorder,
                split=split,
                **paths
            )
            total_str = ""
            if DS == EuroSATIterableLMDBDataset:
                for i, (x, y) in enumerate(ds):
                    total_str += _hash(x) + _hash(y)
                    if i >= len(sample_indices):
                        break
            else:
                for i in sample_indices:
                    x, y = ds[i]
                    total_str += _hash(x) + _hash(y)

            # check timing
            t0 = time.time()
            for i, _ in enumerate(iter(ds)):
                if i >= timing_samples:
                    break
            ds_type = "IterableLMDB " if DS == EuroSATIterableLMDBDataset \
                else "IndexableTif " if DS == EuroSATIndexableTifDataset \
                else "IndexableLMDB"
            print(f"{split}-{ds_type}: {_hash(total_str)} @ {time.time() - t0:.2f}s")

    print()
    for ds_type in ['indexable_lmdb', 'indexable_tif', 'iterable_lmdb']:
        # seed the dataloaders for reproducibility
        torch.manual_seed(seed)
        dm = EuroSATDataModule(
            batch_size=1,
            num_workers=0,
            bandorder=bandorder,
            ds_type=ds_type,
            lmdb_path=lmdb_path,
            metadata_parquet_path=metadata_parquet_path,
            base_path=tif_base_path
        )
        dm.setup()
        total_str = ""
        for i in range(num_batches):
            for x, y in dm.train_dataloader():
                total_str += _hash(x) + _hash(y)
                break
            for x, y in dm.val_dataloader():
                total_str += _hash(x) + _hash(y)
                break
            for x, y in dm.test_dataloader():
                total_str += _hash(x) + _hash(y)
                break
        print(f"datamodule-{ds_type:<14}: {_hash(total_str)}")

if __name__ == "__main__":
    from tqdm import tqdm
    for version in ["rgb", "ms"]:
        for split in ['train', 'validation', 'test', None]:
            print(f"Hashing EuroSAT {version} {split}")
            bandorder = ["B04", "B03", "B02"] if version == "rgb" else ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
            ds = EuroSATIndexableLMDBDataset(
                lmdb_path="../../untracked-files/_reference/EuroSAT.lmdb",
                metadata_parquet_path="../../untracked-files/_reference/EuroSAT.parquet",
                bandorder=bandorder,
                split=split
            )
            hashes = []
            for i in tqdm(range(len(ds)), desc="Hashing EuroSAT"):
                img, label = ds[i]
                # calculate hash of image and label
                img_hash = _hash(img)
                label_hash = _hash(label)
                hashes.append((img_hash, label_hash))
            # save hashes to file
            with open(f"hashes_ES_{version}_{split}.txt", "w") as f:
                for img_hash, label_hash in hashes:
                    f.write(f"{img_hash} {label_hash}\n")
