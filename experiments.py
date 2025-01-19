import argparse

from lightning.pytorch import Trainer, ModelCheckpoint

from base import BaseModel

import customCNN

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

def experiments():
    args = parser.parse_args()

    network = customCNN.get_network(args.arch_name, args.num_channels, args.num_classes, args.pretrained)
    datamodule = None
    model = BaseModel(args, datamodule, network)

    checkpoint_callback = ModelCheckpoint(
        monitor=model.best_metric,
        dirpath="untracked-files",
        filename=f"{args.arch_name}-{args.task}-{{epoch:d}}-{{{model.best_metric}:.3f}}"
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        devices=[0],
        enable_checkpointing=True,
        max_epochs=args.epochs,
    )

    # training
    trainer.fit(model,datamodule)

    best_model_path = trainer.checkpoint_callback.best_model_path # can also be called without trainer.
    best_model = BaseModel.load_from_checkpoint(best_model_path)


if __name__ == "__main__":
    experiments()
