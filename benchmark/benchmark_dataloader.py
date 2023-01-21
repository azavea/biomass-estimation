from os.path import join
import argparse
import time

from tqdm import tqdm
import pandas as pd

from biomass.dataset import BiomassDataModule


def main(args):
    batch_sz = 16
    nb_batches = 50
    rows = []

    def benchmark(num_workers):
        transforms = None
        if not args.no_transforms:
            transforms = [
                {'name': 'sentinel_band_normalize'},
                {'name': 'select_bands', 'bands': 'all'},
                {'name': 'aggregate_months', 'agg_fn': 'mean'}]
        dm = BiomassDataModule(
            root_dir=args.root_dir, batch_size=batch_sz, num_workers=num_workers,
            dataset_limit=batch_sz*nb_batches, use_best_month=True,
            month_items=True, transforms=transforms,
            no_labels=args.no_labels, shuffle_train=not args.no_shuffle_train,
            pin_memory=not args.no_pin_memory,
            use_benchmark_dataset=args.use_benchmark_dataset)
        dm.setup()
        ds = dm.train_dataset
        dl = dm.train_dataloader()

        start = time.time()
        for batch_ind, batch in tqdm(enumerate(dl), total=len(dl)):
            pass
        img_per_sec = 1.0 / ((time.time() - start) / len(ds))
        return img_per_sec

    for num_workers in range(args.max_workers + 1):
        img_per_sec = benchmark(num_workers)
        rows.append([num_workers, img_per_sec])

    df = pd.DataFrame(rows, columns=['num_workers', 'img_per_sec'])
    df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/biomass/')
    parser.add_argument('--no_labels', action='store_true')
    parser.add_argument('--no_transforms', action='store_true')
    parser.add_argument('--max_workers', type=int, default=4)
    parser.add_argument('--no_shuffle_train', action='store_true')
    parser.add_argument('--no_pin_memory', action='store_true')
    parser.add_argument('--use_benchmark_dataset', action='store_true')
    parser.add_argument('--output_path', type=str, default='results.csv')
    args = parser.parse_args()
    main(args)
