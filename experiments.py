import argparse

from lightning.pytorch import Trainer

from base import BaseModel

from models import get_network
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from data.data_BEN import BENDataModule
from data.data_EuroSAT import EuroSATDataModule
from data.caltech101 import Caltech101DataModule
from lightning.pytorch.loggers import WandbLogger

parser = argparse.ArgumentParser(prog='APP4RS', description='Run Experiments.')

parser.add_argument('--task', type=str, default='slc')
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

#added
parser.add_argument('--max_lr', type=float)
parser.add_argument('--pct_start', type=float)
parser.add_argument('--patience', type=int)
    
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
                "lmdb_path": "./untracked-files/caltech101",
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
        'apply_flip': args.apply_flip
    }

    # Initialize datamodule
    datamodule_config = dataset_configs[args.dataset]
    datamodule_config["kwargs"]["augmentation_flags"] = augmentation_flags
    datamodule = datamodule_config["module"](**datamodule_config["kwargs"]) # BenDataModule(args)
    
    # Initialize network
    network = get_network(
        arch_name=args.arch_name,
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        drop_rate=0.3 if args.dropout else 0.0
    )
    
    # Adjust learning rate for Caltech-101
    if args.dataset == "Caltech-101":
        args.learning_rate = 0.025 #TODO: Check if this is true for all experiments. And add to args.
    
    # Initialize model
    model = BaseModel(args, datamodule, network)
    
    checkpoint_callback = ModelCheckpoint(
        monitor=model.best_validation_metric,
        dirpath="untracked-files",
        filename=f"{args.arch_name}-{args.task}-{{epoch:d}}-{{{model.best_validation_metric}:.3f}}"
    )

    early_stopping_callback = EarlyStopping(
        monitor=model.best_validation_metric,
        patience=args.patience,
        mode="max"
    )

    wandb_logger = WandbLogger(
        project="milestone3",
        log_model=True,
        offline=True
    )
    
    trainer = Trainer(
        callbacks=[checkpoint_callback,early_stopping_callback],
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
        
    #TODO reusing same trainer should be no problem right?
    trainer.test(best_model,datamodule)

if __name__ == "__main__":
    experiments()
