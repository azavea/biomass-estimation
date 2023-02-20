from os.path import join
import argparse
import time
import gc

from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp

from biomass.dataset import BiomassDataset, BiomassDataModule


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15, 1, 1)

    def forward(self, x):
        return self.conv(x)


def make_model(encoder_name, in_channels):
    if encoder_name == 'linear':
        model = LinearModel()
    else:
        model = smp.Unet(
            encoder_name=encoder_name, in_channels=in_channels, encoder_weights=None,
            classes=1, activation=None)
    return model


class ModelWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]

        z = self.model(x)
        loss = self.loss_fn(z.squeeze(), y.squeeze())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        return optimizer


def benchmark(format, raw_pytorch, synthetic, encoder_name, batch_size, precision, args):
    print(f'format: {format}')
    print(f'raw_pytorch: {raw_pytorch}')
    print(f'synthetic: {synthetic}')
    print(f'encoder_name: {encoder_name}')
    print(f'batch_size: {batch_size}')
    print(f'precision: {precision}')

    nb_batches = 25
    dataset_size = batch_size * nb_batches

    if synthetic:
        data = torch.randn(dataset_size, 15, 256, 256)
        targets = torch.randn(dataset_size, 1, 256, 256)
        dataset = torch.utils.data.TensorDataset(data, targets)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=8)
    else:
        transforms = [
            {'name': 'sentinel_band_normalize'},
            {'name': 'select_bands', 'bands': 'all'},
            {'name': 'aggregate_months', 'agg_fn': 'mean'}
        ]
        dm = BiomassDataModule(
            root_dir='/mnt/biomass', train_ratio=0.8, batch_size=batch_size, num_workers=8,
            dataset_limit=dataset_size,
            use_best_month=True, month_items=True, transforms=transforms,
            jpg_format=format=='jpg')
        dm.setup()
        dataset = dm.train_dataset
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    _model = make_model(encoder_name, 15)
    model = ModelWrapper(_model)

    if raw_pytorch:
        _model = _model.to('cuda')
        optimizer = torch.optim.Adam(model.parameters())

    gc.collect()
    torch.cuda.empty_cache()

    for phase in ['burnin', 'benchmark']:
        torch.cuda.synchronize()

        if raw_pytorch:
            if phase == 'benchmark':
                start = time.time()

            for batch_ind, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                x = batch[0]
                y = batch[1]
                x = x.to('cuda')
                y = y.to('cuda')
                z = _model(x)
                loss = model.loss_fn(z.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        else:
            trainer = pl.Trainer(gpus=1, precision=precision, max_epochs=1, profiler=args.profiler)
            if phase == 'benchmark':
                start = time.time()
            trainer.fit(model=model, train_dataloaders=train_loader)
    torch.cuda.synchronize()
    dt = time.time() - start
    img_per_sec = len(dataset) / dt
    ms_per_img = (dt / len(dataset)) * 1000

    return img_per_sec, ms_per_img


def main(args):
    rows = []

    for format in args.formats:
        for raw_pytorch in args.raw_pytorch:
            for synthetic in args.synthetic:
                for encoder_name in args.encoder_names:
                    for batch_sz in args.batch_sizes:
                        for precision in args.precisions:
                            for run in range(args.num_runs):
                                img_per_sec, ms_per_img = benchmark(
                                    format, raw_pytorch, synthetic, encoder_name,
                                    batch_sz, precision, args)
                                rows.append([
                                    format, raw_pytorch, synthetic, encoder_name,
                                    batch_sz, precision,
                                    run, round(img_per_sec), round(ms_per_img, 1)])

    df = pd.DataFrame(
        rows,
        columns=['format', 'raw_pytorch', 'synthetic', 'encoder_name', 'batch_sz',
                 'precision', 'run', 'img_per_sec', 'ms_per_img'])
    df.to_csv(args.output_path, index=False)
    print(df)


if __name__ == '__main__':
    encoder_names = ['linear', 'resnet18', 'resnet50']

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='results.csv')
    parser.add_argument('--synthetic', type=str, nargs='+', default=['False', 'True'])
    parser.add_argument('--encoder_names', type=str, nargs='+', default=encoder_names)
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[32])
    parser.add_argument('--precisions', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--raw_pytorch', type=str, nargs='+', default=['False'])
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--formats', type=str, nargs='+', default=['original', 'jpg'])
    parser.add_argument('--profiler', type=str, default=None)


    args = parser.parse_args()

    for encoder_name in args.encoder_names:
        assert encoder_name in encoder_names
    def convert_to_bool(s):
        assert s in ['True', 'False']
        return True if s == 'True' else False
    args.raw_pytorch = [convert_to_bool(s) for s in args.raw_pytorch]
    args.synthetic = [convert_to_bool(s) for s in args.synthetic]
    main(args)
