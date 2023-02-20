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


def main(args):
    rows = []

    start_ind = args.start_ind
    end_ind = args.start_ind + args.nb_items

    for library in args.library:
        if library == 'rasterio':
            img_fns = os.listdir(args.img_dir)
            for phase in range(args.phases):
                start = time.time()
                for img_fn in img_fns[start_ind:end_ind]:
                    with rasterio.open(join(args.img_dir, img_fn)) as ds:
                        ds.read()
                img_per_sec = args.nb_items / (time.time() - start)
                rows.append(['rasterio', phase, img_per_sec])
        elif library == 'skimage':
            img_fns = os.listdir(args.img_dir)
            for phase in range(args.phases):
                start = time.time()
                for img_fn in img_fns[start_ind:end_ind]:
                    x = skio.imread(join(args.img_dir, img_fn))
                img_per_sec = args.nb_items / (time.time() - start)
                rows.append(['skimage', phase, img_per_sec])

    df = pd.DataFrame(rows, columns=['library', 'phase', 'img_per_sec'])
    df.to_csv(args.output_path, index=False)
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/mnt/biomass/train_features')
    parser.add_argument('--start_ind', type=int, default=0)
    parser.add_argument('--nb_items', type=int, default=100)
    parser.add_argument('--library', type=str, nargs='+', default=['skimage', 'rasterio'])
    parser.add_argument('--phases', type=int, default=2)
    parser.add_argument('--output_path', type=str, default='results.csv')

    args = parser.parse_args()
    main(args)
