from os.path import join
import argparse
import time

from tqdm import tqdm
import pandas as pd
import torch
import segmentation_models_pytorch as smp


def main(args):
    rows = []
    nb_batches = 50
    batch_szs = args.batch_szs

    def benchmark(device, batch_sz):
        model = smp.Unet(
            encoder_name=args.encoder_name,
            in_channels=15, encoder_weights=None,
            classes=1, activation=None)
        model = model.to(device)

        start = time.time()
        for _ in tqdm(range(nb_batches)):
            x = torch.randn(batch_sz, 15, 256, 256)
            y = model(x.to(device))
            z = torch.randn_like(y).to(device)
            loss = torch.nn.functional.mse_loss(y, z)
            loss.backward()
        img_per_sec = 1.0 / ((time.time() - start) / (nb_batches * batch_sz))
        return img_per_sec

    for device in ['cpu', 'mps', 'cuda']:
        if device == 'cuda' and not torch.cuda.is_available():
            continue
        elif device == 'mps' and not torch.backends.mps.is_available():
            continue
        for batch_sz in batch_szs:
            img_per_sec = benchmark(device, batch_sz)
            rows.append([device, batch_sz, img_per_sec])

    df = pd.DataFrame(rows, columns=['device', 'batch_sz', 'img_per_sec'])
    df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='results.csv')
    parser.add_argument('--batch_szs', type=int, nargs='+', default=[4, 16])
    parser.add_argument('--encoder_name', type=str, default='resnet18')
    args = parser.parse_args()
    main(args)
