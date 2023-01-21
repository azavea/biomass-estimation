from os.path import join
import argparse
import time
import os

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import rasterio
import skimage.io as skio

from biomass.dataset import BiomassMetadata


@profile
def main(args):
    # using rasterio
    img_dir = args.subset_dir
    img_fns = os.listdir(img_dir)
    for img_fn in img_fns[args.subset_ind:args.subset_ind+args.nb_items]:
        with rasterio.open(join(img_dir, img_fn)) as ds:
            ds.read()

    img_dir = args.full_dir
    img_fns = os.listdir(img_dir)
    for img_fn in img_fns[args.full_ind:args.full_ind+args.nb_items]:
        with rasterio.open(join(img_dir, img_fn)) as ds:
            ds.read()

    # using pillow
    img_dir = args.subset_dir
    img_fns = os.listdir(img_dir)
    for img_fn in img_fns[args.subset_ind:args.subset_ind+args.nb_items]:
        x = skio.imread(join(img_dir, img_fn))

    img_dir = args.full_dir
    img_fns = os.listdir(img_dir)
    for img_fn in img_fns[args.full_ind:args.full_ind+args.nb_items]:
        x = skio.imread(join(img_dir, img_fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_dir', type=str, default='/mnt/biomass/train_features_1000')
    parser.add_argument('--subset_ind', type=int, default=0)
    parser.add_argument('--full_dir', type=str, default='/mnt/biomass/train_features')
    parser.add_argument('--full_ind', type=int, default=0)
    parser.add_argument('--nb_items', type=int, default=100)
    args = parser.parse_args()
    main(args)
