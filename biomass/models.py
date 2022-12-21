from argparse import ArgumentParser
from typing import Any, Dict, cast
from os.path import join

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio


class LinearPixelRegression(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv1(x)


class S1MeanWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x[:, :, 0:4, :, :].mean(1)
        return self.model(x)


def avg_rmse(y, z):
    batch_sz = y.shape[0]
    mse = torch.square(y - z).reshape(batch_sz, -1).mean(dim=1)
    return torch.mean(torch.sqrt(mse))


class TemporalPixelRegression(pl.LightningModule):
    def config_task(self) -> None:
        if self.hparams["model_name"] == "s1_linear_regression":
            self.model = S1MeanWrapper(LinearPixelRegression(4, 1))
        else:
            raise ValueError(
                f"Model type '{self.hparams['model_name']}' is not valid. "
                f"Currently, only supports 's1_linear_regression'."
            )
        self.example_input_array = torch.Tensor(1, 12, 15, 256, 256)

    def __init__(self, model_name, learning_rate, learning_rate_schedule_patience) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config_task()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, chip_metadata = batch
        z = self.forward(x)
        loss = avg_rmse(y, z)
        log_dict = {'train_loss': loss}
        self.log_dict(
            log_dict, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, chip_metadata = batch
        z = self.forward(x)
        loss = avg_rmse(y, z)
        log_dict = {'val_loss': loss}
        self.log_dict(
            log_dict, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

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
                dst.write(_prediction[0], 1)



