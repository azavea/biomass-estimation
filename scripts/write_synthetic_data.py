import os
from os.path import join
import argparse

import numpy as np
import rasterio


def main(args):
    nb_arrs = 1000
    batch_sz = 10

    numpy_dir = join(args.output_dir, 'numpy')
    numpy_batch_dir = join(args.output_dir, 'numpy-batch')
    tif_dir = join(args.output_dir, 'tif')

    for dir in [numpy_dir, tif_dir, numpy_batch_dir]:
        os.makedirs(dir, exist_ok=True)

    for ind in range(nb_arrs):
        arr = np.random.randint(0, 255, (15, 256, 256), dtype=np.uint8)
        np.save(join(numpy_dir, f'random_{ind}'), arr)
        with rasterio.open(join(tif_dir, f'random_{ind}.tif'), 'w', driver='GTiff',
                           height=256, width=256, count=15, dtype=rasterio.uint8) as dst:
            dst.write(arr)

    for ind in range(nb_arrs // batch_sz):
        arr = np.random.randint(0, 255, (batch_sz, 15, 256, 256), dtype=np.uint8)
        np.save(join(numpy_batch_dir, f'random_{ind}'), arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/mnt/biomass/synthetic-data/')
    args = parser.parse_args()
    main(args)
