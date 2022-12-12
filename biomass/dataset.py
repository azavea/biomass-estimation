import os
from os.path import join, isfile, basename
import random
from multiprocessing.pool import ThreadPool
import warnings

from tqdm.autonotebook import tqdm
import s3fs
import matplotlib.pyplot as plt
import matplotlib
import pandas
import numpy as np
import rasterio
from torch.utils.data import Dataset
import torch
import torch.nn as nn

us_s3_uri = 's3://drivendata-competition-biomassters-public-us'
fs = s3fs.S3FileSystem(anon=True)


class BiomassMetadata():
    def __init__(self, features_path, labels_path):
        self.features_df = pandas.read_csv(features_path)
        self.labels_df = pandas.read_csv(labels_path)

        # I'm not totally sure this is the right order for S1
        self.s1_bands = ['VV Asc', 'VH Asc', 'VV Desc', 'VH Desc']
        self.s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'CLP']
        self.bands = self.s1_bands + self.s2_bands
        self.months = [
            'September', 'October', 'November', 'December', 'January', 'February',
            'March', 'April', 'May', 'June', 'July', 'August']

    def get_image_uri(self, split, chip_id, satellite, month, root_uri=us_s3_uri):
        return join(root_uri, f'{split}_features/{chip_id}_{satellite}_{month:02}.tif')

    def get_label_uri(self, split, chip_id, root_uri=us_s3_uri):
        if split != 'train':
            raise Exception('split must be train')
        return join(root_uri, f'{split}_agbm/{chip_id}_agbm.tif')

    def get_chip_ids(self, split):
        chip_ids = self.features_df.query(f'split=="{split}"').chip_id.unique()
        random.seed(1234)
        random.shuffle(chip_ids)
        return chip_ids


class BiomassDataset(Dataset):
    def __init__(self, root_dir, split, chip_ids=None, transform=None, download=False):
        if split not in ['train', 'test']:
            raise Exception(f'{split} is not a valid split')

        self.root_dir = root_dir
        self.features_metadata_path = join(root_dir, 'features_metadata.csv')
        self.labels_metadata_path = join(root_dir, 'train_agbm_metadata.csv')
        self.image_dir = join(root_dir, f'{split}_features')
        self.label_dir = join(root_dir, 'train_agbm')

        self.split = split
        self.chip_ids = chip_ids
        self.transform = transform

        if download:
            self.download_data()
        else:
            self.metadata = BiomassMetadata(
                self.features_metadata_path, self.labels_metadata_path)
            self.chip_ids = (chip_ids if chip_ids is not None else
                    self.metadata.get_chip_ids(split))

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, ind):
        chip_id = self.chip_ids[ind]
        full_months = torch.zeros(12, 2)
        partial_months = torch.zeros(12, 2)
        month_arrs = []
        s1_masks = []
        s2_masks = []

        for month in range(0, 12):
            sat_arrs = []
            for satellite in ['S1', 'S2']:
                image_path = self.metadata.get_image_uri(
                    self.split, chip_id, satellite, month, root_uri=self.root_dir)

                if satellite == 'S1':
                    # S1 is float32 and -9999 means missing data
                    if isfile(image_path):
                        with rasterio.open(image_path) as img:
                            sat_arr = torch.from_numpy(img.read())
                            partial_months[month, 0] = 1
                        s1_mask = torch.sum(sat_arr != -9999, dim=0) == 4
                    else:
                        sat_arr = torch.zeros(4, 256, 256)
                        s1_mask = torch.zeros(256, 256)
                    s1_masks.append(s1_mask)
                    if torch.all(s1_mask):
                        full_months[month, 0] = 1
                else:
                    if isfile(image_path):
                        with rasterio.open(image_path) as img:
                            # S2 is uint16 and the last band is cloud probability
                            # (ranges 0-100, or 255 for noise)
                            sat_arr = torch.from_numpy(img.read().astype(np.float32))
                            sat_arr[0:-1] = sat_arr[0:-1]
                            sat_arr[-1] = sat_arr[-1]
                            partial_months[month, 1] = 1
                            s2_mask = sat_arr[-1] != 255
                    else:
                        sat_arr = torch.zeros(11, 256, 256)
                        s2_mask = torch.zeros(256, 256)
                    s2_masks.append(s2_mask)
                    if torch.all(s2_mask):
                        full_months[month, 1] = 1
                sat_arrs.append(sat_arr)
            month_arrs.append(torch.cat(sat_arrs))

        x = torch.stack(month_arrs)
        y = None
        if self.split == 'train':
            label_path = self.metadata.get_label_uri(
                self.split, chip_id, root_uri=self.root_dir)
            # labels are float32 with zero used for missing data and real values
            with rasterio.open(label_path) as label:
                y = torch.from_numpy(label.read()).squeeze()

        chip_metadata = {
            'chip_id': chip_id,
            'full_months': full_months,
            'partial_months': partial_months,
            's1_masks': torch.stack(s1_masks),
            's2_masks': torch.stack(s2_masks),
        }
        if self.transform:
            x, y = self.transform(x, y)
        return x, y, chip_metadata

    def plot_sample(self, x, y, chip_metadata, z=None, out_path=None):
        nrows = x.shape[0] + 1
        ncols = 17
        s1_masks = chip_metadata['s1_masks']
        s2_masks = chip_metadata['s2_masks']
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
                        ax.set_ylabel(self.metadata.months[row_ind])
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

    def download_data(self):
        # TODO more robust check of whether data is downloaded
        if os.path.isfile(self.features_metadata_path):
            warnings.warn('Skipping dataset download because it appears unnecessary.')
            return

        fs.download(
            join(us_s3_uri, 'features_metadata.csv'), self.features_metadata_path)
        fs.download(
            join(us_s3_uri, 'train_agbm_metadata.csv'), self.labels_metadata_path)
        self.metadata = BiomassMetadata(
            self.features_metadata_path, self.labels_metadata_path)
        self.chip_ids = (
            self.chip_ids if self.chip_ids is not None else
            self.metadata.get_chip_ids(self.split))

        os.makedirs(self.image_dir)
        os.makedirs(self.label_dir)

        download_tasks = []
        for chip_id in self.chip_ids:
            for satellite in ['S1', 'S2']:
                for month in range(0, 12):
                    image_uri = self.metadata.get_image_uri('train', chip_id, satellite, month)
                    image_fn = basename(image_uri)
                    image_path = join(self.image_dir, image_fn)
                    download_tasks.append((image_uri, image_path))
            label_uri = self.metadata.get_label_uri('train', chip_id)
            label_path = join(self.label_dir, basename(label_uri))
            download_tasks.append((label_uri, label_path))

        def download_file(x):
            from_uri, to_path = x
            if fs.exists(from_uri):
                fs.download(from_uri, to_path)

        pool = ThreadPool(8)
        for _ in tqdm(pool.imap_unordered(download_file, download_tasks),
                        total=len(download_tasks)):
            pass


class BiomassBandNormalize():
    def __call__(self, x, y):
        x = x.clone()
        s1 = x[:, 0:4, :, :]
        s2 = x[:, 4:14, :, :]
        clp = x[:, 14:15, :, :]

        s1[s1 == -9999] = -50
        s1 += 50
        s1 /= 50

        s2 /= 10_000

        clp[clp == 255] = 100
        clp /= 100

        x = torch.cat([s1, s2, clp], dim=1)
        if y is not None:
            y = torch.clamp(y, 0, 400)

        return x, y
