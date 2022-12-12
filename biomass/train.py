from argparse import ArgumentParser
from typing import Any
from os.path import join, basename, isfile
import os
import shutil

import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import seaborn as sns

from biomass.dataset import BiomassDataset, BiomassBandNormalize
from biomass.models import (
    TemporalPixelRegressionLightning, S1MeanWrapper,
    LinearPixelRegression, avg_rmse)


def cli_main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TemporalPixelRegressionLightning.add_model_specific_args(parser)
    args = parser.parse_args()

    root_dir = '/Users/lewfish/data/biomass/'
    dataset_dir = join(root_dir, 'dataset')
    output_dir = join(root_dir, 'output')
    lightning_dir = join(output_dir, 'lightning')
    shutil.rmtree(lightning_dir)
    os.makedirs(lightning_dir, exist_ok=True)

    # Use a small sample for testing. Assumes this was downloaded by the
    # explore_data.ipynb notebook
    chip_ids = ['d8e45923', 'a4529dfd', 'fa341f98', 'b27bcdad', '284543b9',
                '329f5682', '52b1d478', 'a7cf91c5', '174ca2b3', '25a25054']
    train_chip_ids = chip_ids[0:6]
    val_chip_ids = chip_ids[6:]
    train_ds = BiomassDataset(
        dataset_dir, 'train', train_chip_ids, transform=BiomassBandNormalize())
    # split is 'train' just because the sample of chip_ids are all from the training set.
    val_ds = BiomassDataset(
        dataset_dir, 'train', val_chip_ids, transform=BiomassBandNormalize())

    batch_size = 2
    num_workers = 0
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    model = TemporalPixelRegressionLightning(
        S1MeanWrapper(LinearPixelRegression(4, 1)),
        lr=args.lr)

    csv_logger = CSVLogger(join(lightning_dir, 'csv-logs'))
    trainer = pl.Trainer.from_argparse_args(
        args, min_epochs=1, max_epochs=args.epochs+1, logger=[csv_logger])
    trainer.fit(model, train_dl, val_dl)

    model_path = join(lightning_dir, 'model.ckpt')
    trainer.save_checkpoint(model_path)


if __name__ == '__main__':
    cli_main()
