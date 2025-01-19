import argparse

from lightning.pytorch import Trainer

from base import BaseModel

from models import get_network
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data.data_BEN import BENDataModule
from data.data_EuroSAT import EuroSATDataModule
from data.caltech101 import Caltech101DataModule

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


#added
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--patience', type=int)

def experiments():
    args = parser.parse_args()

    network = get_network(arch_name=args.arch_name, num_channels=args.num_channels, num_classes=args.num_classes, pretrained=args.pretrained)

    #TODO which bandorder? caltech probably rgb?
    bandorder = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]
    #bandorder = ["B04", "B03", "B02"] if version == "rgb" else ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    if(args.dataset == "tiny-BEN"):
        datamodule = BENDataModule(batch_size=args.batch_size,num_workers=args.num_workers,bandorder=bandorder,ds_type='indexable_lmdb',lmdb_path=args.lmdb_path,metadata_parquet_path=args.metadata_parquet_path)
    elif(args.dataset == "EuroSAT"):
        datamodule = EuroSATDataModule(batch_size=args.batch_size,num_workers=args.num_workers,bandorder=bandorder,ds_type='indexable_lmdb',lmdb_path=args.lmdb_path,metadata_parquet_path=args.metadata_parquet_path)
    elif(args.dataset == "Caltech-101"):
        datamodule = Caltech101DataModule(lmdb_path="./untracked-files/caltech101",batch_size=args.batch_size,num_workers=args.num_workers)
    model = BaseModel(args, datamodule, network)

    checkpoint_callback = ModelCheckpoint(
        monitor=model.best_metric,
        dirpath="untracked-files",
        filename=f"{args.arch_name}-{args.task}-{{epoch:d}}-{{{model.best_metric}:.3f}}"
    )

    #TODO ignore these parameters: , min_delta=0.00, verbose=False ?
    early_stopping_callback = EarlyStopping(monitor=model.best_metric, patience=args.patience, mode="max")

    trainer = Trainer(
        callbacks=[checkpoint_callback,early_stopping_callback],
        accelerator='gpu',
        devices=[0],
        enable_checkpointing=True,
        max_epochs=args.epochs,
    )

    # training
    trainer.fit(model,datamodule)

    best_model_path = trainer.checkpoint_callback.best_model_path # can also be called without trainer.
    best_model = BaseModel.load_from_checkpoint(best_model_path)

    #TODO reusing same trainer should be no problem right?
    trainer.test(best_model,datamodule)

if __name__ == "__main__":
    experiments()
