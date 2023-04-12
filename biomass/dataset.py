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
    # I'm not totally sure this is the right order for S1
    s1_bands = ['VV Asc', 'VH Asc', 'VV Desc', 'VH Desc']
    s2_bands = [
        'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'CLP']
    bands = s1_bands + s2_bands
    months = [
        'September', 'October', 'November', 'December', 'January', 'February',
        'March', 'April', 'May', 'June', 'July', 'August']
    month2ind = {m: i for i, m in enumerate(months)}

    def __init__(self, features_path, labels_path):
        self.features_df = pandas.read_csv(features_path)
        self.labels_df = pandas.read_csv(labels_path)

    def get_image_uri(self, split, chip_id, satellite, month, root_uri=us_s3_uri):
        return join(root_uri, f'{split}_features/{chip_id}_{satellite}_{month:02}.tif')

    def get_label_uri(self, split, chip_id, root_uri=us_s3_uri):
        return join(root_uri, f'{split}_agbm/{chip_id}_agbm.tif')

    def get_chip_ids(self, split):
        return self.features_df.query(f'split=="{split}"').chip_id.unique()


class BiomassDataset(Dataset):
    def __init__(self, root_dir='data', split='train', use_best_month=False,
                 month_items=False, transform=None, chip_ids=None, jpg_format=False):
        if split not in ['train', 'test']:
            raise Exception(f'{split} is not a valid split')

        self.root_dir = root_dir
        self.jpg_format = jpg_format
        self.features_metadata_path = join(root_dir, 'features_metadata.csv')
        self.labels_metadata_path = join(root_dir, 'train_agbm_metadata.csv')
        self.image_dir = join(root_dir, f'{split}_features')
        self.label_dir = join(root_dir, f'{split}_agbm')

        self.split = split
        self.use_best_month = use_best_month
        self.month_items = month_items
        self.transform = transform
        self.metadata = BiomassMetadata(
            self.features_metadata_path, self.labels_metadata_path)

        suffix = '' if split == 'train' else '_TEST'
        self.best_months_df = pandas.read_csv(
            join(self.root_dir, f'TILE_LIST_BEST_MONTHS{suffix}.csv'))

        split_chip_ids = self.metadata.get_chip_ids(self.split)
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
                    for chip_id in split_chip_ids]

        if chip_ids is not None:
            chip_ids = set(chip_ids)
            self.chip_id_month_pairs = [
                (chip_id, months)
                for chip_id, months in self.chip_id_month_pairs
                if chip_id in chip_ids]

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

    def get_jpg_item(self, ind):
        chip_id, month_inds = self.chip_id_month_pairs[ind]
        img_dir = join(self.root_dir, f'jpg_{self.split}_features')

        month_arrs = []
        for month in month_inds:
            x = torch.from_numpy(skio.imread(join(img_dir, f'{chip_id}_{month:02}.jpg')))
            x = x.to(torch.float32) / 255
            x = x.reshape((15, 256, 256))
            month_arrs.append(x)

        x = torch.stack(month_arrs)
        y = None
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

    def get_jpg_year_item(self, ind):
        chip_id = self.chip_id_month_pairs[ind][0]
        img_dir = join(self.root_dir, f'jpg_year_{self.split}_features')

        x = torch.from_numpy(skio.imread(join(img_dir, f'{chip_id}.jpg')))
        x = x.to(torch.float32) / 255
        x = torch.stack(torch.split(x, 256), dim=0).reshape(12, 15, 256, 256)

        y = None
        if self.split == 'train':
            label_path = self.metadata.get_label_uri(
                self.split, chip_id, root_uri=self.root_dir)
            # labels are float32 with zero used for missing data and real values
            y = torch.from_numpy(skio.imread(label_path)).squeeze()

        chip_metadata = {
            'chip_id': chip_id,
            'months': torch.arange(0, 12),
        }

        if self.transform:
            x, y = self.transform((x, y))

        if y is None:
            return x, chip_metadata
        else:
            return x, y, chip_metadata

    def __getitem__(self, ind):
        if self.jpg_format:
            if self.month_items:
                return self.get_jpg_item(ind)
            else:
                return self.get_jpg_year_item(ind)

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

    @staticmethod
    def plot_sample(x, y, chip_metadata, z=None, out_path=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        month_weights = None
        month_outputs = None
        month_pixel_weights = None
        if z is not None:
            if isinstance(z, dict):
                month_weights = z.get('month_weights')
                month_pixel_weights = z.get('month_pixel_weights')
                month_outputs = z.get('month_outputs')
                z = z['output']

        s1_masks = chip_metadata.get('s1_masks')
        s2_masks = chip_metadata.get('s2_masks')
        months = chip_metadata['months']
        col_names = BiomassMetadata.bands + ['S1 Mask', 'S2 Mask', 'Month Output', 'Attention Mask']
        nrows = x.shape[0] + 2
        ncols = len(col_names)

        fig = plt.figure(
            constrained_layout=True, figsize=(1.5 * ncols, 1.5 * nrows))
        gs = fig.add_gridspec(nrows, ncols)

        def remove_ticks(ax):
            ax.set_xticks([])
            ax.set_yticks([])

        for row_ind in range(x.shape[0]):
            for col_ind in range(ncols):
                ax = fig.add_subplot(gs[row_ind, col_ind])

                _x = None
                vmin, vmax = None, None
                if col_ind < 15:
                    _x = x[row_ind, col_ind, :, :]
                elif col_ind == 15 and s1_masks is not None:
                    _x = s1_masks[row_ind]
                elif col_ind == 16 and s2_masks is not None:
                    _x = s2_masks[row_ind]
                elif col_ind == 17 and month_outputs is not None:
                    _x = month_outputs[row_ind]
                elif col_ind == 18 and month_pixel_weights is not None:
                    _x = month_pixel_weights[row_ind]
                    vmin, vmax = 0, 1

                if _x is not None:
                    ax.imshow(_x, vmin=vmin, vmax=vmax)

                if col_ind == 0:
                    month = BiomassMetadata.months[months[row_ind]]
                    label = (
                        month if month_weights is None
                        else f'{month}: ({month_weights[row_ind]:.2f})')
                    ax.set_ylabel(label)

                if row_ind == 0:
                    ax.set_title(col_names[col_ind])
                remove_ticks(ax)

        # plot the labels and other chip-wide information in the last 2 rows
        if y is not None:
            ax = fig.add_subplot(gs[x.shape[0]:x.shape[0]+2, 0:2])
            ax.imshow(y, vmin=0, vmax=400)
            ax.set_title('Biomass GT')
            remove_ticks(ax)
        if z is not None:
            ax = fig.add_subplot(gs[x.shape[0]:x.shape[0]+2, 2:4])
            ax.imshow(z, vmin=0, vmax=400)
            ax.set_title('Biomass Prediction')
            remove_ticks(ax)
        if y is not None and z is not None:
            nb_samples = 1000
            rand_indices = np.random.choice(
                y.reshape(-1).shape[0], (nb_samples,), replace=False)
            y_sample = y.reshape(-1)[rand_indices]
            z_sample = z.reshape(-1)[rand_indices]
            ax = fig.add_subplot(gs[x.shape[0]:x.shape[0]+2, 4:6])
            ax.scatter(y_sample, z_sample, alpha=0.2)
            max_val = max(y_sample.max(), z_sample.max())
            ax.plot([0, max_val], [0, max_val])
            ax.set_xlabel('GT')
            ax.set_ylabel('Prediction')
        if month_weights is not None:
            ax = fig.add_subplot(gs[x.shape[0]:x.shape[0]+2, 6:8])
            ax.bar(np.arange(12), month_weights)
            ax.set_xlabel('Month')
            ax.set_ylabel('Attention')

        if out_path:
            plt.savefig(
                out_path, bbox_inches='tight', pad_inches=0.2, transparent=False,
                dpi=150)
        else:
            plt.show()


# TODO move data normalization and augmentation to GPU
class BiomassDataModule(pl.LightningDataModule):
    def __init__(
            self, root_dir='data', train_ratio=0.8, batch_size=8,
            num_workers=8, dataset_limit=None, use_best_month=False,
            month_items=False, transforms=None,
            shuffle_train=True, pin_memory=True, jpg_format=False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_limit = dataset_limit
        self.use_best_month = use_best_month
        self.month_items = month_items
        self.transforms = transforms
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        self.jpg_format = jpg_format

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
            month_items=self.month_items, transform=transform,
            jpg_format=self.jpg_format)
        if self.dataset_limit is not None and self.dataset_limit < len(all_train_dataset):
            all_train_dataset = Subset(all_train_dataset, list(range(self.dataset_limit)))
        train_sz = round(len(all_train_dataset) * self.train_ratio)
        val_sz = len(all_train_dataset) - train_sz
        self.train_dataset, self.val_dataset = random_split(
            all_train_dataset, [train_sz, val_sz])
        self.predict_dataset = BiomassDataset(
            root_dir=self.root_dir, split='test', use_best_month=self.use_best_month,
            month_items=self.month_items, transform=transform, jpg_format=self.jpg_format)
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
            persistent_workers=True,
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
            persistent_workers=True,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )
