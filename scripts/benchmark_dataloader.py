from os.path import join
import argparse
import time

from tqdm import tqdm
import pandas as pd

from biomass.dataset import BiomassDataModule


def main(args):
    batch_size = 32
    nb_batches = 25
    rows = []

    def benchmark(format, num_workers):
        transforms = [
            {
                'name': 'sentinel_band_normalize',
                'norm_x': format == 'original',
                'norm_y': True,
            },
            {'name': 'select_bands', 'bands': 'all'},
            {'name': 'aggregate_months', 'agg_fn': 'mean'}
        ]
        dm = BiomassDataModule(
            root_dir=args.root_dir, train_ratio=0.8, batch_size=batch_size,
            dataset_limit=batch_size*nb_batches,
            num_workers=num_workers, use_best_month=True, month_items=True,
            transforms=transforms, jpg_format=format == 'jpg')
        dm.setup()
        ds = dm.train_dataset
        dl = dm.train_dataloader()

        start = time.time()
        for batch_ind, batch in tqdm(enumerate(dl), total=len(dl)):
            x, y, _ = batch
            x.to('cuda')
            y.to('cuda')
        img_per_sec = len(ds) / (time.time() - start)
        return img_per_sec

    for format in args.formats:
        for num_workers in [0, 1, 2, 4, 8]:
            img_per_sec = benchmark(format, num_workers)
            rows.append([format, num_workers, img_per_sec])

    df = pd.DataFrame(rows, columns=['format', 'num_workers', 'img_per_sec'])
    df.to_csv(args.output_path, index=False)
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/biomass/')
    parser.add_argument('--formats', type=str, nargs='+', default=['original', 'jpg'])
    parser.add_argument('--output_path', type=str, default='results.csv')
    args = parser.parse_args()
    main(args)
