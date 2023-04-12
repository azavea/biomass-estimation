from argparse import ArgumentParser
from typing import Any, Dict, cast
from os.path import join
import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, OneCycleLR
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
    def __init__(self, in_channels, encoder_name='resnet18', agg_method='month_mean',
                 time_embed_method=None, encoder_weights=None):
        super().__init__()

        if agg_method not in ['month_mean', 'month_attn', 'month_pixel_attn']:
            raise ValueError(f'agg_method={agg_method} is not a valid aggregation method')
        if time_embed_method not in [None, 'add_inputs']:
            raise ValueError(
                f'time_embed_method={time_embed_method} is not '
                'a valid time embedding method')
        self.agg_method = agg_method
        self.time_embed_method = time_embed_method

        aux_params = None
        classes = 1
        if agg_method == 'month_attn':
            aux_params = {'classes': 1, 'activation': None}
        elif agg_method == 'month_pixel_attn':
            classes = 2

        if time_embed_method == 'add_inputs':
            in_channels += 2
        self.unet = smp.Unet(
            encoder_name=encoder_name, in_channels=in_channels,
            encoder_weights=encoder_weights,
            classes=classes, activation=None, aux_params=aux_params)

        x = (torch.arange(12) / 12) * 2 * math.pi
        self.embeds = torch.vstack((torch.sin(x), torch.cos(x))).T
        assert self.embeds.shape == (12, 2)

    def forward(self, x):
        batch_sz, times, channels, height, width = x.shape
        month_pixel_weights = None

        if self.time_embed_method == 'add_inputs':
            # Add month embedding as input channels
            embeds = self.embeds.to(x.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            assert embeds.shape == (1, times, 2, 1, 1)
            embeds = embeds.expand(batch_sz, -1, -1, height, width)
            assert embeds.shape == (batch_sz, times, 2, height, width)
            x = torch.cat((x, embeds), dim=2)
            assert x.shape == (batch_sz, times, channels + 2, height, width)
            channels += 2

        if self.agg_method == 'month_mean':
            z = self.unet(x.reshape(-1, *x.shape[2:]))
            assert z.shape == (batch_sz * times, 1, height, width)
            z = z.reshape(*x.shape[0:2], 1, *x.shape[3:])
            assert z.shape == (batch_sz, times, 1, height, width)
            month_outputs = z.squeeze(2)
            assert month_outputs.shape == (batch_sz, times, height, width)
            z = z.mean(dim=1)
            assert z.shape == (batch_sz, 1, height, width)
            month_weights = torch.full((batch_sz, times), 1.0 / times, device=x.device)
            assert month_weights.shape == (batch_sz, times)
        elif self.agg_method == 'month_attn':
            z, logits = self.unet(x.reshape(-1, channels, height, width))
            assert z.shape == (batch_sz * times, 1, height, width)
            assert logits.shape == (batch_sz * times, 1)

            z = z.reshape(batch_sz, times, height, width)
            month_outputs = z
            assert z.shape == (batch_sz, times, height, width)

            logits = logits.reshape(batch_sz, times)
            assert logits.shape == (batch_sz, times)
            probs = torch.softmax(logits, dim=1)
            month_weights = probs
            assert probs.shape == (batch_sz, times)
            probs = probs.unsqueeze(-1).unsqueeze(-1)
            assert probs.shape == (batch_sz, times, 1, 1)

            z = (z * probs).sum(dim=1)
            assert z.shape == (batch_sz, height, width)
        elif self.agg_method == 'month_pixel_attn':
            out = self.unet(x.reshape(-1, channels, height, width))
            assert out.shape == (batch_sz * times, 2, height, width)
            z = out[:, 0, :, :]
            assert z.shape == (batch_sz * times, height, width)
            logits = out[:, 1, :, :]
            assert logits.shape == (batch_sz * times, height, width)
            z = z.reshape(batch_sz, times, height, width)
            month_outputs = z
            assert z.shape == (batch_sz, times, height, width)
            logits = logits.reshape(batch_sz, times, height, width)
            assert logits.shape == (batch_sz, times, height, width)

            month_pixel_weights = torch.softmax(logits, dim=1)
            assert month_pixel_weights.shape == (batch_sz, times, height, width)
            month_weights = month_pixel_weights.mean(dim=(2, 3))
            assert month_weights.shape == (batch_sz, times)

            z = (z * month_pixel_weights).sum(dim=1)
            assert z.shape == (batch_sz, height, width)
        return {
            'output': z, 'month_weights': month_weights, 'month_outputs': month_outputs,
            'month_pixel_weights': month_pixel_weights
        }


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
        self.month_loss = self.hparams['month_loss']
        self.month_loss_scale = self.hparams['month_loss_scale']

    def __init__(self, model_name, learning_rate, lr_scheduler,
                 loss, month_loss, month_loss_scale, model_args):
        super().__init__()
        self.save_hyperparameters()
        self.config_task()

    def forward(self, x):
        return self.model(x)

    def step(self, split, batch, batch_idx):
        x, y, chip_metadata = batch
        batch_sz, times, channels, height, width = x.shape
        assert y.shape == (batch_sz, height, width)

        z = self.forward(x)

        month_outputs = None
        if isinstance(z, dict):
            month_outputs = z['month_outputs']
            z = z['output']
        if self.month_loss and month_outputs is None:
            raise ValueError('month_loss is True but month_outputs is None.')

        assert z.shape == y.shape
        mae = torch.nn.functional.l1_loss(y, z)
        rmse = avg_rmse(y, z)
        loss_fn = torch.nn.functional.l1_loss if self.loss == 'mae' else avg_rmse
        loss = loss_fn(y, z)
        if self.month_loss:
            # Make sure the month outputs are the same shape as y
            _y = y.unsqueeze(1)
            assert _y.shape == (batch_sz, 1, height, width)
            assert month_outputs.shape == (batch_sz, 12, height, width)
            loss = loss + self.month_loss_scale * loss_fn(_y, month_outputs)

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

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        lr = self.hparams["learning_rate"]
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)

        scheduler = None
        interval = 'step'
        lr_scheduler = self.hparams.get('lr_scheduler')
        if lr_scheduler.get('name') == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=lr_scheduler['learning_rate_schedule_patience'],
                threshold=lr_scheduler['learning_rate_schedule_threshold'],
            )
            interval = 'epoch'
        elif lr_scheduler.get('name') == 'CyclicLR':
            train_ds_sz = len(self.trainer.datamodule.train_dataset)
            steps_per_epoch = max(1, train_ds_sz // self.trainer.datamodule.batch_size)
            total_steps = self.trainer.max_epochs * steps_per_epoch
            step_size_up = (self.trainer.max_epochs // 2) * steps_per_epoch
            step_size_down = total_steps - step_size_up
            scheduler = CyclicLR(
                optimizer,
                base_lr=lr / 10,
                max_lr=lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                cycle_momentum=False,)
        elif lr_scheduler.get('name') == 'OneCycleLR':
            train_ds_sz = len(self.trainer.datamodule.train_dataset)
            steps_per_epoch = max(1, train_ds_sz // self.trainer.datamodule.batch_size)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'val_loss',
                'interval': interval,
                'frequency': 1
            },
        }


class BiomassPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch,
                           batch_idx, dataloader_idx):
        if isinstance(prediction, dict):
            prediction = prediction['output']
        chip_ids = batch[1]['chip_id']

        for chip_id, _prediction in zip(chip_ids, prediction):
            out_path = join(self.output_dir, f'{chip_id}_agbm.tif')

            with rasterio.open(out_path, 'w', dtype=rasterio.float32, count=1,
                               height=_prediction.shape[0],
                               width=_prediction.shape[1]) as dst:
                dst.write(_prediction.cpu().numpy(), 1)
