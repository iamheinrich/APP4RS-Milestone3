# partial functions
from functools import partial
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


def _hash(data):
    return md5(str(data).encode()).hexdigest()


BEN_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]
BEN_CLASSES.sort()
assert len(BEN_CLASSES) == 19, f"Expected 19 classes, got {len(BEN_CLASSES)}"

BEN_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]


def _stringlist_to_tensor(stringlist: List[str]) -> torch.Tensor:
    tensor = torch.zeros(len(BEN_CLASSES))
    for s in stringlist:
        tensor[BEN_CLASSES.index(s)] = 1
    return tensor


class BENIndexableLMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the BigEarthNet dataset using an lmdb file.

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
        self.keys = self.metadata['patch_id'].tolist()
        # sort keys to ensure reproducibility
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def _open_lmdb(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)

    def _stack(self, patch):
        # interpolate each channel to 120x120
        patch = {
            k: torch.nn.functional.interpolate(
                torch.from_numpy(v).float().unsqueeze(0),
                size=(120, 120),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            for k, v in patch.items()
        }
        # stack channels
        patch = torch.stack([patch[k] for k in self.bandorder], dim=0)
        return patch

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
        patch = self._stack(patch)
        if self.transform:
            patch = self.transform(patch)

        # get label from metadata
        label = self.metadata[self.metadata['patch_id'] == key]['labels']
        assert len(label) == 1
        label = label.values[0]
        label = _stringlist_to_tensor(label)
        return patch, label


class BENIndexableTifDataset(Dataset):
    def __init__(self, base_path: str, bandorder: List, split=None, transform=None):
        """
        Dataset for the BigEarthNet dataset using tif files.

        :param base_path: path to the source BigEarthNet dataset (root of the tar file)
        :param bandorder: names of the bands to use, e.g. ["B04", "B03", "B02"] for RGB
        :param split: split of the dataset to use, one of 'train', 'validation', 'test', None (uses all data)
        :param transform: a torchvision transform to apply to the images after loading
        """
        self.tif_path = base_path + "/BigEarthNet-Lithuania-Summer-S2/"
        self.metadata_parquet_path = base_path + "/lithuania_summer.parquet"
        self.transform = transform
        self.bandorder = bandorder
        self.metadata = pd.read_parquet(self.metadata_parquet_path)
        if split is not None:
            self.metadata = self.metadata[self.metadata['split'] == split]
        self.keys = self.metadata['patch_id'].tolist()
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
        # get img data from tif
        patch_path = self.tif_path + key
        patch_path = "_".join(patch_path.split('_')[:-2]) + f"/{key}"
        patch = []
        for band in self.bandorder:
            band_data = rasterio.open(f"{patch_path}/{key}_{band}.tif").read()
            # interpolate to 120x120
            band_data = torch.nn.functional.interpolate(
                torch.from_numpy(band_data).float().unsqueeze(0),
                size=(120, 120),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            patch.append(band_data)
        patch = torch.stack(patch, dim=0)
        if self.transform:
            patch = self.transform(patch)

        # get label from metadata
        label = self.metadata[self.metadata['patch_id'] == key]['labels']
        assert len(label) == 1
        label = label.values[0]
        label = _stringlist_to_tensor(label)
        return patch, label


class BENIterableLMDBDataset(IterableDataset):
    def __init__(self, lmdb_path: str, metadata_parquet_path: str, bandorder: List, split=None, transform=None,
                 with_keys=False):
        """
        IterableDataset for the BigEarthNet dataset using an lmdb file.

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
        self.env = None
        self.env_iter = None
        self.metadata = pd.read_parquet(metadata_parquet_path)
        self.split = split
        if split is not None:
            self.metadata = self.metadata[self.metadata['split'] == split]
        self.keys = self.metadata['patch_id'].tolist()
        # sort keys to ensure reproducibility
        self.keys.sort()
        self.key_start = 0
        self.key_end = len(self.keys)
        self.initialized = None

    def _stack(self, patch):
        # interpolate each channel to 120x120
        patch = {
            k: torch.nn.functional.interpolate(
                torch.from_numpy(v).float().unsqueeze(0),
                size=(120, 120),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            for k, v in patch.items()
        }
        # stack channels
        patch = torch.stack([patch[k] for k in self.bandorder], dim=0)
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
                label = self.metadata[self.metadata['patch_id'] == key]['labels']
                assert len(label) == 1
                label = label.values[0]
                label = _stringlist_to_tensor(label)
                yield patch, label


class BENDataModule(LightningDataModule):
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
        DataModule for the BigEarthNet dataset.

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
            self.dataset = partial(BENIndexableTifDataset,
                                   base_path=base_path,
                                   bandorder=bandorder)
        elif ds_type == 'iterable_lmdb':
            assert lmdb_path is not None, 'lmdb_path must be provided for iterable_lmdb dataset'
            assert metadata_parquet_path is not None, \
                'metadata_parquet_path must be provided for iterable_lmdb dataset'
            self.dataset = partial(BENIterableLMDBDataset,
                                   lmdb_path=lmdb_path,
                                   metadata_parquet_path=metadata_parquet_path,
                                   bandorder=bandorder)
        elif ds_type == 'indexable_lmdb':
            assert lmdb_path is not None, 'lmdb_path must be provided for indexable_lmdb dataset'
            assert metadata_parquet_path is not None, \
                'metadata_parquet_path must be provided for indexable_lmdb dataset'
            self.dataset = partial(BENIndexableLMDBDataset,
                                   lmdb_path=lmdb_path,
                                   metadata_parquet_path=metadata_parquet_path,
                                   bandorder=bandorder)
        else:
            raise ValueError(f'Unknown dataset type: {ds_type}')
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset(
                split='train'
            )
            self.val_dataset = self.dataset(
                split='validation'
            )
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset(
                split='test'
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
    Test the BigEarthNet dataset classes.

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
        for DS in [BENIndexableLMDBDataset, BENIndexableTifDataset, BENIterableLMDBDataset]:
            paths = {
                "base_path": tif_base_path,
            } if DS == BENIndexableTifDataset else {
                "lmdb_path": lmdb_path,
                "metadata_parquet_path": metadata_parquet_path,
            }
            # create dataset
            ds = DS(
                bandorder=bandorder,
                split=split,
                **paths
            )
            # check values of sample_indices, collect hashes
            total_str = ""
            if DS == BENIterableLMDBDataset:
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
            ds_type = "IterableLMDB " if DS == BENIterableLMDBDataset \
                else "IndexableTif " if DS == BENIndexableTifDataset \
                else "IndexableLMDB"
            print(f"{split}-{ds_type}: {_hash(total_str)} @ {time.time() - t0:.2f}s")

    print()
    for ds_type in ['indexable_lmdb', 'indexable_tif', 'iterable_lmdb']:
        # seed the dataloaders for reproducibility
        torch.manual_seed(seed)
        dm = BENDataModule(
            batch_size=1,
            num_workers=0,
            bandorder=bandorder,
            ds_type=ds_type,
            lmdb_path=lmdb_path,
            metadata_parquet_path=metadata_parquet_path,
            base_path=tif_base_path
        )
        dm.setup()
        # again, collect hashes
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
            print(f"Hashing BEN {version} {split}")
            bandorder = ["B04", "B03", "B02"] if version == "rgb" else ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
            ds = BENIndexableLMDBDataset(
                lmdb_path="../../untracked-files/_reference/BigEarthNet.lmdb",
                metadata_parquet_path="../../untracked-files/_reference/BigEarthNet.parquet",
                bandorder=bandorder,
                split=split
            )
            hashes = []
            for i in tqdm(range(len(ds)), desc="Hashing BEN"):
                img, label = ds[i]
                # calculate hash of image and label
                img_hash = _hash(img)
                label_hash = _hash(label)
                hashes.append((img_hash, label_hash))
            # save hashes to file
            with open(f"hashes_BEN_{version}_{split}.txt", "w") as f:
                for img_hash, label_hash in hashes:
                    f.write(f"{img_hash} {label_hash}\n")