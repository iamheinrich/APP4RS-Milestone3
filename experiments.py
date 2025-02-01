import wandb
wandb.login(key="d12c4aa89f6e2fb545cabd2314cca6c865e382d2")
import argparse

from lightning.pytorch import Trainer

from base import BaseModel

from models import get_network
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from data.data_BEN import BENDataModule, BENIndexableLMDBDataset
from data.data_EuroSAT import EuroSATDataModule, EuroSATIndexableLMDBDataset
from data.caltech101 import Caltech101DataModule, Caltech101Dataset
from lightning.pytorch.loggers import WandbLogger

import torch
from utils import compute_channel_statistics_rs, compute_channel_statistics_rgb
from data.transform import get_remote_sensing_transform, get_caltech_transform

parser = argparse.ArgumentParser(prog='APP4RS', description='Run Experiments.')

parser.add_argument('--task', type=str)
parser.add_argument('--logging_dir', type=str)
    
parser.add_argument('--dataset', type=str)
parser.add_argument('--lmdb_path', type=str)
parser.add_argument('--metadata_parquet_path', type=str)

parser.add_argument('--num_channels', type=int)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--num_workers', type=int)
parser.add_argument('--batch_size', type=int)
    
parser.add_argument('--arch_name', type=str)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--epochs', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--weight_decay', type=float)
    
# Augmentation arguments
parser.add_argument('--apply_random_resize_crop', action='store_true')
parser.add_argument('--apply_cutout', action='store_true')
parser.add_argument('--apply_brightness', action='store_true')
parser.add_argument('--apply_contrast', action='store_true')
parser.add_argument('--apply_grayscale', action='store_true')
parser.add_argument('--apply_sharpen', action='store_true')
parser.add_argument('--apply_flip', action='store_true')
parser.add_argument('--apply_resize112', action='store_true')

#added
parser.add_argument('--max_lr', type=float)
parser.add_argument('--pct_start', type=float)
parser.add_argument('--patience', type=int)

#latest
parser.add_argument('--early_stopping', action='store_true')

#for wandb project argument
parser.add_argument('--experiment_type', type=str)
    
def experiments():
    args = parser.parse_args()

    # Define datasets and respective configs
    dataset_configs = {
        "tiny-BEN": {
            "module": BENDataModule,
            "kwargs": {
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "bandorder": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"], # According to bandorder in reference dataloader
                "ds_type": 'indexable_lmdb',
                "lmdb_path": args.lmdb_path,
                "metadata_parquet_path": args.metadata_parquet_path
            }
        },
        "EuroSAT": {
            "module": EuroSATDataModule,
            "kwargs": {
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "bandorder": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"], # According to bandorder in reference dataloader
                "ds_type": 'indexable_lmdb',
                "lmdb_path": args.lmdb_path,
                "metadata_parquet_path": args.metadata_parquet_path
            }
        },
        "Caltech-101": {
            "module": Caltech101DataModule,
            "kwargs": {
                "lmdb_path": args.lmdb_path,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers
                #TODO: Add bandorder?
            }
        }
    }
    
    # Configure dataset
    if args.dataset not in dataset_configs:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Set up augmentation flags based on args
    augmentation_flags = {
        'apply_random_resize_crop': args.apply_random_resize_crop,
        'apply_cutout': args.apply_cutout,
        'apply_brightness': args.apply_brightness,
        'apply_contrast': args.apply_contrast,
        'apply_grayscale': args.apply_grayscale,
        'apply_sharpen': args.apply_sharpen,
        'apply_flip': args.apply_flip,
        'apply_resize112': args.apply_resize112
    }

    # Initialize datamodule
    datamodule_config = dataset_configs[args.dataset]
    #datamodule_config["kwargs"]["augmentation_flags"] = augmentation_flags 

    #EUROSET
    #CALTECH
    #END BOTH
    
    if (args.dataset=="tiny-BEN") or (args.dataset=="EuroSAT"):
        dataset_class = EuroSATIndexableLMDBDataset if args.dataset=="EuroSAT" else BENIndexableLMDBDataset
        tmp_train_dataset = dataset_class(
                lmdb_path=args.lmdb_path,
                metadata_parquet_path=dataset_configs[args.dataset]["kwargs"]["metadata_parquet_path"],
                bandorder=dataset_configs[args.dataset]["kwargs"]["bandorder"],
                split='train',
                transform=None
        )
    else :
        tmp_train_dataset = Caltech101Dataset(
                dataset_path=args.lmdb_path,
                split='train',
                transform=None
        )

    # Create a temporary dataloader to compute statistics
    temp_train_dataloader = torch.utils.data.DataLoader(
        tmp_train_dataset,
        batch_size=dataset_configs[args.dataset]["kwargs"]["batch_size"],
        num_workers=dataset_configs[args.dataset]["kwargs"]["num_workers"],
        shuffle=False  # No need to shuffle for statistics
    )    

    if (args.dataset=="tiny-BEN") or (args.dataset=="EuroSAT"):
        # Compute statistics using only training data to prevent leakage
        mean, std, percentile = compute_channel_statistics_rs(temp_train_dataloader)
        train_transform = get_remote_sensing_transform(percentile,mean,std,
                                                 args.apply_random_resize_crop,
                                                 args.apply_cutout,
                                                 args.apply_brightness,
                                                 args.apply_contrast,
                                                 args.apply_grayscale,
                                                 args.apply_sharpen,
                                                 args.apply_flip,
                                                 args.apply_resize112)
        val_test_transform = get_remote_sensing_transform(percentile,mean,std,
                                                 apply_resize112=args.apply_resize112)
    else:
        # Compute statistics using only training data
        mean, std = compute_channel_statistics_rgb(temp_train_dataloader)
        train_transform = get_caltech_transform(mean,std,
                                                 args.apply_random_resize_crop,
                                                 args.apply_cutout,
                                                 args.apply_brightness,
                                                 args.apply_contrast,
                                                 args.apply_grayscale,
                                                 args.apply_sharpen)
        val_test_transform = get_caltech_transform(mean,std)

    datamodule = datamodule_config["module"](**datamodule_config["kwargs"],
                                             train_transform = train_transform,
                                             val_test_transform =val_test_transform) # BenDataModule(args)
    
    # Initialize network
    network = get_network(
        arch_name=args.arch_name,
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        drop_rate=0.3 if args.dropout else 0.0
    )
    
    # Initialize model
    model = BaseModel(args, datamodule, network)
    
    checkpoint_callback = ModelCheckpoint(
        monitor=model.best_validation_metric,
        dirpath="untracked-files",
        filename=f"{args.arch_name}-{args.task}-{{epoch:d}}-{{{model.best_validation_metric}:.3f}}"
    )

    if args.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor=model.best_validation_metric,
            patience=args.patience,
            mode="max"
        )

    if args.experiment_type == "multi_model_benchmark":
        pretrained_str = "pretrained" if args.pretrained else "retrain"
        dropout_str = "dropout" if args.dropout else "dropless"
        run_name = f"{args.arch_name}_{pretrained_str}_{dropout_str}"
    elif args.experiment_type == "augmentation_study":
        if args.apply_random_resize_crop:
            run_name = "apply_random_resize_crop"
        elif args.apply_cutout:
            run_name = "apply_cutout"
        elif args.apply_brightness:
            run_name = "apply_brightness"
        elif args.apply_contrast:
            run_name = "apply_contrast"
        elif args.apply_grayscale:
            run_name = "apply_grayscale"
    elif args.experiment_type == "feature_extraction_study":
        run_name = "tsne_resnet18_eurosat"
    else:
        raise NotImplementedError(f"This args.experiment_type:{args.experiment_type} is not implemented in experiments.py")

    wandb_logger = WandbLogger(
        project=args.experiment_type,
        group=args.dataset,
        name=run_name,
        log_model=True,
        config = vars(args),
        offline=False
    )
    
    print(wandb_logger.run.id,wandb_logger.run.name)

    trainer = Trainer(
        callbacks=[checkpoint_callback,early_stopping_callback] if args.early_stopping else [checkpoint_callback],
        logger=wandb_logger,
        accelerator='auto',
        devices='auto',
        enable_checkpointing=True,
        max_epochs=args.epochs,
    )
    
    # training
    trainer.fit(model,datamodule)
    
    best_model_path = trainer.checkpoint_callback.best_model_path # can also be called without trainer.
    best_model = BaseModel.load_from_checkpoint(checkpoint_path=best_model_path, args=args, datamodule=datamodule, network=network)
        
    trainer.test(best_model,datamodule)

if __name__ == "__main__":
    experiments()
