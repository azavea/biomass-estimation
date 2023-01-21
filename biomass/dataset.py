from typing import Any, Dict, Optional
import os
from os.path import join, isfile, basename
import random
from multiprocessing.pool import ThreadPool
import warnings
import logging

from tqdm.autonotebook import tqdm
import s3fs
import matplotlib.pyplot as plt
import matplotlib
import pandas
import numpy as np
from torch.utils.data import (
    Dataset, DataLoader, Subset, random_split)
import torch
import torch.nn as nn
import pytorch_lightning as pl
import skimage.io as skio

from biomass.transforms import build_transform

us_s3_uri = 's3://drivendata-competition-biomassters-public-us'
fs = s3fs.S3FileSystem(anon=True)


class BiomassMetadata():
    def __init__(self, features_path, labels_path):
        self.features_df = pandas.read_csv(features_path)
        self.labels_df = pandas.read_csv(labels_path)

        # I'm not totally sure this is the right order for S1
        self.s1_bands = ['VV Asc', 'VH Asc', 'VV Desc', 'VH Desc']
        self.s2_bands = [
            'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'CLP']
        self.bands = self.s1_bands + self.s2_bands
        self.months = [
            'September', 'October', 'November', 'December', 'January', 'February',
            'March', 'April', 'May', 'June', 'July', 'August']
        self.month2ind = {m: i for i, m in enumerate(self.months)}

    def get_image_uri(self, split, chip_id, satellite, month, root_uri=us_s3_uri):
        return join(root_uri, f'{split}_features/{chip_id}_{satellite}_{month:02}.tif')

    def get_label_uri(self, split, chip_id, root_uri=us_s3_uri):
        return join(root_uri, f'{split}_agbm/{chip_id}_agbm.tif')

    def get_chip_ids(self, split):
        return self.features_df.query(f'split=="{split}"').chip_id.unique()


class BiomassDataset(Dataset):
    def __init__(self, root_dir='data', split='train', use_best_month=False,
                 month_items=False, transform=None, no_labels=False):
        if split not in ['train', 'test']:
            raise Exception(f'{split} is not a valid split')

        self.root_dir = root_dir
        self.features_metadata_path = join(root_dir, 'features_metadata.csv')
        self.labels_metadata_path = join(root_dir, 'train_agbm_metadata.csv')
        self.image_dir = join(root_dir, f'{split}_features')
        self.label_dir = join(root_dir, f'{split}_agbm')

        self.split = split
        self.transform = transform
        self.no_labels = no_labels
        self.metadata = BiomassMetadata(
            self.features_metadata_path, self.labels_metadata_path)

        suffix = '' if split == 'train' else '_TEST'
        self.best_months_df = pandas.read_csv(
            join(self.root_dir, f'TILE_LIST_BEST_MONTHS{suffix}.csv'))

        self.chip_ids = self.metadata.get_chip_ids(self.split)
        self.chip_id_month_pairs = []

        if use_best_month:
            if not month_items:
                raise ValueError('use_best_month=True requires month_items=True')
            self.chip_id_month_pairs = [
                (chip_id, [month])
                for chip_id, month in zip(self.best_months_df.chipid, self.best_months_df.month)]
        else:
            all_month_inds = list(range(12))
            df = self.metadata.features_df.query(f'split=="{split}" and satellite=="S1"')
            if month_items:
                self.chip_id_month_pairs = [
                    (chip_id, [self.metadata.month2ind[month]])
                    for chip_id, month in zip(df.chip_id, df.month)]
            else:
                self.chip_id_month_pairs = [
                    (chip_id, all_month_inds)
                    for chip_id in self.chip_ids]

    def __len__(self):
        return len(self.chip_id_month_pairs)

    def get_s1_img(self, img_path):
        # S1 is float32 and -9999 means missing data
        partial = 0
        full = 0
        if isfile(img_path):
            img_arr = torch.from_numpy(
                skio.imread(img_path)).permute(2, 0, 1)
            partial = 1
            mask = (torch.sum(img_arr != -9999, dim=0) == 4).float()
        else:
            img_arr = torch.zeros(4, 256, 256)
            mask = torch.zeros(256, 256)
        if torch.all(mask):
            full = 1
        return img_arr, mask, partial, full

    def get_s2_img(self, img_path):
        partial = 0
        full = 0
        if isfile(img_path):
            # S2 is uint16 and the last band is cloud probability
            # (ranges 0-100, or 255 for noise)
            img_arr = torch.from_numpy(
                skio.imread(img_path).astype(np.float32)).permute(2, 0, 1)
            partial = 1
            mask = (img_arr[-1] != 255).float()
        else:
            img_arr = torch.zeros(11, 256, 256)
            mask = torch.zeros(256, 256)
        if torch.all(mask):
            full = 1
        return img_arr, mask, partial, full

    def download_data(self, chip_ids):
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        download_tasks = []
        for chip_id in chip_ids:
            for satellite in ['S1', 'S2']:
                for month in range(0, 12):
                    image_uri = self.metadata.get_image_uri(
                        self.split, chip_id, satellite, month)
                    image_fn = basename(image_uri)
                    image_path = join(self.image_dir, image_fn)
                    download_tasks.append((image_uri, image_path))
            label_uri = self.metadata.get_label_uri(self.split, chip_id)
            label_path = join(self.label_dir, basename(label_uri))
            download_tasks.append((label_uri, label_path))

        def download_file(x):
            from_uri, to_path = x
            if fs.exists(from_uri):
                fs.download(from_uri, to_path)

        for task in tqdm(download_tasks, desc='Downloading dataset'):
            download_file(task)

    def getitem_nometadata(self, ind):
        chip_id, month_inds = self.chip_id_month_pairs[ind]

        month_arrs = []

        for month in month_inds:
            img_path = self.metadata.get_image_uri(
                self.split, chip_id, 'S1', month, root_uri=self.root_dir)
            if isfile(img_path):
                img_arr = torch.from_numpy(skio.imread(img_path)).permute(2, 0, 1)
            else:
                img_arr = torch.zeros(4, 256, 256)
            s1_arr = img_arr

            img_path = self.metadata.get_image_uri(
                self.split, chip_id, 'S2', month, root_uri=self.root_dir)
            if isfile(img_path):
                # S2 is uint16 and the last band is cloud probability
                # (ranges 0-100, or 255 for noise)
                img_arr = torch.from_numpy(
                    skio.imread(img_path).astype(np.float32)).permute(2, 0, 1)
            else:
                img_arr = torch.zeros(11, 256, 256)
            s2_arr = img_arr

            month_arrs.append(torch.cat([s1_arr, s2_arr]))

        x = torch.stack(month_arrs)

        y = None
        if not self.no_labels:
            if self.split == 'train':
                label_path = self.metadata.get_label_uri(
                    self.split, chip_id, root_uri=self.root_dir)
                # labels are float32 with zero used for missing data and real values
                y = torch.from_numpy(skio.imread(label_path)).squeeze()

        chip_metadata = {
            'chip_id': chip_id,
            'months': torch.tensor(month_inds),
        }

        if self.transform:
            x, y = self.transform((x, y))

        if y is None:
            return x, chip_metadata
        else:
            return x, y, chip_metadata

    def __getitem__(self, ind):
        chip_id, month_inds = self.chip_id_month_pairs[ind]

        full_months = []
        partial_months = []
        month_arrs = []
        s1_masks = []
        s2_masks = []

        for month in month_inds:
            image_path = self.metadata.get_image_uri(
                self.split, chip_id, 'S1', month, root_uri=self.root_dir)
            s1_arr, s1_mask, s1_partial, s1_full = self.get_s1_img(image_path)
            image_path = self.metadata.get_image_uri(
                self.split, chip_id, 'S2', month, root_uri=self.root_dir)
            s2_arr, s2_mask, s2_partial, s2_full = self.get_s2_img(image_path)

            month_arrs.append(torch.cat([s1_arr, s2_arr]))
            partial_months.append(torch.tensor([s1_partial, s2_partial]))
            full_months.append(torch.tensor([s1_full, s2_full]))
            s1_masks.append(s1_mask)
            s2_masks.append(s2_mask)

        x = torch.stack(month_arrs)
        y = None
        if self.split == 'train':
            label_path = self.metadata.get_label_uri(
                self.split, chip_id, root_uri=self.root_dir)
            # labels are float32 with zero used for missing data and real values
            y = torch.from_numpy(skio.imread(label_path)).squeeze()

        chip_metadata = {
            'chip_id': chip_id,
            'full_months': torch.stack(full_months),
            'partial_months': torch.stack(partial_months),
            's1_masks': torch.stack(s1_masks),
            's2_masks': torch.stack(s2_masks),
            'months': torch.tensor(month_inds),
        }

        if self.transform:
            x, y = self.transform((x, y))

        if y is None:
            return x, chip_metadata
        else:
            return x, y, chip_metadata

    def plot_sample(self, x, y, chip_metadata, z=None, out_path=None):
        nrows = x.shape[0] + 1
        ncols = 17
        s1_masks = chip_metadata['s1_masks']
        s2_masks = chip_metadata['s2_masks']
        months = chip_metadata['months']
        col_names = self.metadata.bands + ['S1 Mask', 'S2 Mask']

        fig, axs = plt.subplots(
            nrows, ncols, constrained_layout=True, figsize=(1.5 * ncols, 1.5 * nrows),
            squeeze=False)

        for row_ind, row_axs in enumerate(axs):
            if row_ind == nrows - 1:
                # plot the label in the last row
                for col_ind, ax in enumerate(row_axs):
                    if y is not None and col_ind == 0:
                        ax.imshow(y)
                        ax.set_title('Biomass GT')
                    if z is not None and col_ind == 1:
                        ax.imshow(z)
                        ax.set_title('Biomass Prediction')
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                for col_ind, ax in enumerate(row_axs):
                    # Add S1 and S2 masks to the last columns.
                    if col_ind == 15:
                        _x = s1_masks[row_ind]
                    elif col_ind == 16:
                        _x = s2_masks[row_ind]
                    else:
                        _x = x[row_ind, col_ind, :, :]

                    ax.imshow(_x)

                    if col_ind == 0:
                        ax.set_ylabel(self.metadata.months[months[row_ind]])
                    if row_ind == 0:
                        ax.set_title(col_names[col_ind])

                    ax.set_xticks([])
                    ax.set_yticks([])

        if out_path:
            plt.savefig(
                out_path, bbox_inches='tight', pad_inches=0.2, transparent=False,
                dpi=300)
        else:
            plt.show()


# TODO move data normalization and augmentation to GPU
class BiomassDataModule(pl.LightningDataModule):
    def __init__(
            self, root_dir='data', train_ratio=0.8, batch_size=8,
            num_workers=8, dataset_limit=None, use_best_month=False,
            month_items=False, transforms=None, no_labels=False,
            shuffle_train=True, pin_memory=True) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_limit = dataset_limit
        self.use_best_month = use_best_month
        self.month_items = month_items
        self.transforms = transforms
        self.no_labels = no_labels
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transform = (
            None if self.transforms is None else build_transform(self.transforms))
        all_train_dataset = BiomassDataset(
            root_dir=self.root_dir, split='train', use_best_month=self.use_best_month,
            month_items=self.month_items, transform=transform, no_labels=self.no_labels)
        if self.dataset_limit is not None and self.dataset_limit < len(all_train_dataset):
            all_train_dataset = Subset(all_train_dataset, list(range(self.dataset_limit)))
        self.train_dataset, self.val_dataset = random_split(
            all_train_dataset, [self.train_ratio, 1-self.train_ratio])
        self.predict_dataset = BiomassDataset(
            root_dir=self.root_dir, split='test', use_best_month=self.use_best_month,
            month_items=self.month_items, transform=transform)
        if self.dataset_limit is not None and self.dataset_limit < len(self.predict_dataset):
            self.predict_dataset = Subset(
                self.predict_dataset, list(range(self.dataset_limit)))

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.
        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
