from os.path import join
import argparse
import os
import shutil


def main(args):
    os.makedirs(args.output_dir)

    img_dir = join(args.root_dir, 'train_features')
    img_fns = os.listdir(img_dir)
    img_fns.sort()
    for img_fn in img_fns[0:args.nb_files]:
        shutil.copyfile(join(img_dir, img_fn), join(args.output_dir, img_fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/biomass/')
    parser.add_argument('--output_dir', type=str, default='/mnt/biomass/train_features_subset')
    parser.add_argument('--nb_files', type=int, default=0)
    args = parser.parse_args()
    main(args)
