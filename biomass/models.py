from argparse import ArgumentParser
from typing import Any, Dict, cast
from os.path import join

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio
import segmentation_models_pytorch as smp


class PixelLinearRegression(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv1(x)


class PixelMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 1))

    def forward(self, x):
        return self.mlp(x)


class TemporalUNet(nn.Module):
    def __init__(self, in_channels, encoder_name='resnet18', agg_method='month_mean'):
        super().__init__()
        self.agg_method = agg_method
        aux_params = (
            {'classes': 1, 'activation': None}
            if agg_method == 'month_attn' else None)
        self.unet = smp.Unet(
            encoder_name=encoder_name, in_channels=in_channels, encoder_weights=None,
            classes=1, activation=None, aux_params=aux_params)

    def forward(self, x):
        # x: (batch_sz, time, channels, height, width)
        if self.agg_method == 'month_mean':
            z = self.unet(x.reshape(-1, *x.shape[2:]))
            # z: (batch_sz * time, 1, height, width)
            z = z.reshape(*x.shape[0:2], 1, *x.shape[3:])
            # z: (batch_sz, time, 1, height, width)
            z = z.mean(dim=1)
            # z: (batch_sz, 1, height, width)
        elif self.agg_method == 'month_attn':
            z, logits = self.unet(x.reshape(-1, *x.shape[2:]))
            # z: (batch_sz * time, 1, height, width)
            # logits: (batch_sz * time, 1)
            z = z.reshape(*x.shape[0:2], 1, *x.shape[3:])
            logits = logits.reshape(*x.shape[0:2], 1)
            # z: (batch_sz, time, 1, height, width)
            # logits: (batch_sz, time, 1)
            probs = torch.softmax(logits, dim=1)
            # probs: (batch_sz, time, 1)
            probs = probs.unsqueeze(-1).unsqueeze(-1)
            # probs: (batch_sz, time, 1, 1, 1)
            z = (z * probs).sum(dim=1)
            # z: (batch_sz, 1, height, width)
        else:
            raise ValueError(f"Aggregation method '{self.agg_method}' is not valid.")
        return z


def avg_rmse(y, z):
    batch_sz = y.shape[0]
    mse = torch.square(y - z).reshape(batch_sz, -1).mean(dim=1)
    return torch.mean(torch.sqrt(mse))


class TemporalPixelRegression(pl.LightningModule):
    def config_task(self) -> None:
        # TODO infer in_channels and example_input_array
        in_channels = self.hparams['model_args']['in_channels']
        self.example_input_array = torch.Tensor(1, in_channels, 256, 256)
        if self.hparams['model_name'] == 'pixel_linear_regression':
            self.model = PixelLinearRegression(in_channels, 1)
        elif self.hparams['model_name'] == 'pixel_mlp':
            hidden_channels = self.hparams['model_args']['hidden_channels']
            self.model = PixelMLP(in_channels, hidden_channels, 1)
        elif self.hparams['model_name'] == 'unet':
            self.model = smp.Unet(
                encoder_name=self.hparams['model_args']['encoder_name'],
                in_channels=in_channels, encoder_weights=None,
                classes=1, activation=None)
        elif self.hparams['model_name'] == 'temporal_unet':
            self.example_input_array = torch.Tensor(1, 12, in_channels, 256, 256)
            self.model = TemporalUNet(**self.hparams['model_args'])
        else:
            raise ValueError(
                f"Model type '{self.hparams['model_name']}' is not valid. ")

        loss = self.hparams['loss']
        self.loss = loss
        if loss not in ['rmse', 'mae']:
            raise ValueError(f'{loss} is not valid.')

    def __init__(self, model_name, learning_rate, learning_rate_schedule_patience,
                 learning_rate_schedule_threshold, loss, model_args):
        super().__init__()
        self.save_hyperparameters()
        self.config_task()

    def forward(self, x):
        return self.model(x)

    def step(self, split, batch, batch_idx):
        x, y, chip_metadata = batch
        z = self.forward(x)
        # y is (batch_sz, height, width)
        # ensure z is the same shape if it is (batch_sz, 1, height, width)
        if z.dim() == 4 and z.shape[1] == 1:
            z = z.squeeze(1)
        mae = torch.nn.functional.l1_loss(y, z)
        rmse = avg_rmse(y, z)
        if self.loss == 'mae':
            loss = mae
        elif self.loss == 'rmse':
            loss = rmse
        log_dict = {f'{split}_loss': loss, f'{split}_mae': mae, f'{split}_rmse': rmse}
        self.log_dict(
            log_dict, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self.step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step('val', batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        x, chip_metadata = batch
        return self.forward(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                    threshold=self.hparams["learning_rate_schedule_threshold"],
                ),
                "monitor": "val_loss",
            },
        }


class BiomassPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch,
                           batch_idx, dataloader_idx):
        chip_ids = batch[1]['chip_id']
        for chip_id, _prediction in zip(chip_ids, prediction):
            out_path = join(self.output_dir, f'{chip_id}_agbm.tif')
            with rasterio.open(out_path, 'w', dtype=rasterio.float32, count=1,
                               height=_prediction.shape[1],
                               width=_prediction.shape[2]) as dst:
                dst.write(_prediction[0].cpu().numpy(), 1)
