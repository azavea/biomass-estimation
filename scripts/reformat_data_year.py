from os.path import join
from os import makedirs
import argparse

import numpy as np
import torch
import skimage.io as skio
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm

from biomass.transforms import SentinelBandNormalize
from biomass.dataset import BiomassDataset


def main(args):
    transform = SentinelBandNormalize()

    for split in ['train', 'test']:
        ds = BiomassDataset(
            args.root_dir, split=split, use_best_month=False,
            month_items=False, transform=transform)
        dir_name = f'jpg_year_{split}_features'
        out_dir = join(args.root_dir, dir_name)
        makedirs(out_dir, exist_ok=True)

        def reformat(ind):
            if split == 'test':
                x, metadata = ds[ind]
            else:
                x, _, metadata = ds[ind]

            chip_id = metadata['chip_id']
            x = (x.squeeze().clamp_(0, 1) * 255).to(torch.uint8)
            x = torch.vstack(tuple(x.reshape(-1, 256, 256)))
            skio.imsave(
                join(out_dir, f'{chip_id}.jpg'),
                x.numpy(), quality=90)

        thread_map(reformat, range(len(ds)), max_workers=args.max_workers, chunksize=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/biomass/')
    parser.add_argument('--max_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)
