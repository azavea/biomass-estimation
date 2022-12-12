from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl


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


class TemporalPixelRegressionLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.lr = lr
        self.example_input_array = torch.Tensor(1, 12, 15, 256, 256)

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
        x, y, metadata = batch
        z = self.forward(x)
        loss = avg_rmse(y, z)
        log_dict = {'validation_loss': loss}
        self.log_dict(
            log_dict, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--epochs", type=int, default=2)
        return parser

