from os.path import join
import argparse
import time
import os
import gc

import pandas as pd
import skimage.io as skio
import numpy as np


def main(args):
    rows = []
    batch_sz = 10

    for format in ['numpy', 'tif', 'numpy-batch']:
        img_fns = os.listdir(join(args.img_dir, format))
        for phase in range(args.phases):
            gc.collect()
            start = time.time()
            if format == 'tif':
                for img_fn in img_fns:
                    img = skio.imread(join(args.img_dir, format, img_fn))
            elif format.startswith('numpy'):
                for img_fn in img_fns:
                    x = np.load(join(args.img_dir, format, img_fn), allow_pickle=True)
            dt = time.time() - start
            nb_imgs = len(img_fns) * (batch_sz if format == 'numpy-batch' else 1)
            img_per_sec = nb_imgs / dt
            ms_per_img = (dt / nb_imgs) * 1000
            rows.append([format, phase, round(img_per_sec), round(ms_per_img, 2)])

    df = pd.DataFrame(rows, columns=['format', 'phase', 'img_per_sec', 'ms_per_img'])
    df.to_csv(args.output_path, index=False)
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/mnt/biomass/synthetic-data')
    parser.add_argument('--phases', type=int, default=2)
    parser.add_argument('--output_path', type=str, default='results.csv')

    args = parser.parse_args()
    main(args)
