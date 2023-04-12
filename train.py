# Code in this file was adapted from
# https://github.com/microsoft/torchgeo/blob/main/train.py
import os
from os.path import join
from typing import Any, Dict, Tuple, Type, cast
import warnings
import zipfile
import shutil
import subprocess

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor)
from pytorch_lightning.loggers import WandbLogger
import torch
from tqdm import tqdm

from biomass.models import TemporalPixelRegression, BiomassPredictionWriter
from biomass.dataset import BiomassDataModule, BiomassDataset


def set_up_omegaconf() -> DictConfig:
    conf = OmegaConf.load('conf/defaults.yaml')
    command_line_conf = OmegaConf.from_cli()

    if 'config_file' in command_line_conf:
        config_fn = command_line_conf.config_file
        if not os.path.isfile(config_fn):
            raise FileNotFoundError(f'config_file={config_fn} is not a valid file')

        user_conf = OmegaConf.load(config_fn)
        conf = OmegaConf.merge(conf, user_conf)

    conf = OmegaConf.merge(conf, command_line_conf)
    conf = cast(DictConfig, conf)

    return conf


def main(conf: DictConfig) -> None:
    run_name = conf.program.run_name

    if os.path.isfile(conf.program.output_dir):
        raise NotADirectoryError('`program.output_dir` must be a directory')
    output_dir = conf.program.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if len(os.listdir(output_dir)) > 0:
        if conf.program.overwrite:
            warnings.warn(
                f'WARNING! The run output directory, {output_dir}, already exists, '
                'we might overwrite data in it!')
        else:
            raise FileExistsError(
                f"The run output directory, {output_dir}, already exists and isn't "
                "empty. We don't want to overwrite any existing results, exiting..."
            )

        if conf.program.clear_output_dir:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

    pred_dir = join(conf.program.output_dir, 'test-preds')
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = join(conf.program.output_dir, 'test-preds.zip')

    with open(join(output_dir, 'experiment_config.yaml'), 'w') as f:
        OmegaConf.save(config=conf, f=f)

    task_args = OmegaConf.to_object(conf.experiment.task)
    task = TemporalPixelRegression(**task_args)
    datamodule_args = OmegaConf.to_object(conf.experiment.datamodule)
    datamodule = BiomassDataModule(**datamodule_args)
    datamodule.setup()

    csv_logger = pl_loggers.CSVLogger(conf.program.log_dir, name=run_name)
    wandb_logger = None
    if conf.program.wandb_project:
        wandb_logger = WandbLogger(
            project=conf.program.wandb_project,
            name=run_name)
        wandb_logger.experiment.config.update({
            'trainer': dict(conf.trainer), 'experiment': dict(conf.experiment)})

    monitor_metric = 'val_loss'
    mode = 'min'

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric, dirpath=output_dir, save_top_k=1, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    pred_writer = BiomassPredictionWriter(
        output_dir=pred_dir, write_interval='batch')

    trainer_args = OmegaConf.to_object(conf.trainer)
    trainer_args['callbacks'] = [
        checkpoint_callback, lr_monitor, pred_writer]
    trainer_args['logger'] = [csv_logger]
    if wandb_logger is not None:
        trainer_args['logger'].append(wandb_logger)
    trainer_args['default_root_dir'] = output_dir

    trainer = pl.Trainer(**trainer_args)

    if conf.program.plot_dataset_samples > 0:
        train_ds = datamodule.train_dataset
        val_ds = datamodule.val_dataset
        train_plot_dir = join(output_dir, 'dataset-plots')
        os.makedirs(train_plot_dir, exist_ok=True)
        out_paths = []
        for ind in tqdm(range(conf.program.plot_dataset_samples),
                        desc='Plotting dataset samples'):
            if ind < len(train_ds):
                out_path = join(train_plot_dir, f'train-{ind}.jpg')
                out_paths.append(out_path)
                x, y, chip_metadata = train_ds[ind]
                BiomassDataset.plot_sample(x, y, chip_metadata, out_path=out_path)
            if ind < len(val_ds):
                out_path = join(train_plot_dir, f'val-{ind}.jpg')
                out_paths.append(out_path)
                x, y, chip_metadata = val_ds[ind]
                BiomassDataset.plot_sample(x, y, chip_metadata, out_path=out_path)
        if wandb_logger is not None:
            wandb_logger.log_image(key='dataset-plots', images=out_paths)

    if conf.program.train:
        if trainer_args.get('auto_lr_find'):
            trainer.tune(model=task, datamodule=datamodule)
        trainer.fit(model=task, datamodule=datamodule)

    if conf.program.predict:
        ckpt_path = join(output_dir, 'last.ckpt')
        task = TemporalPixelRegression.load_from_checkpoint(ckpt_path)

        if conf.program.plot_predictions > 0:
            val_ds = datamodule.val_dataset
            pred_plot_dir = join(output_dir, 'prediction-plots')
            out_paths = []
            os.makedirs(pred_plot_dir, exist_ok=True)
            task.eval()
            with torch.no_grad():
                for ind in tqdm(range(conf.program.plot_predictions),
                                desc='Plotting predictions'):
                    if ind < len(val_ds):
                        x, y, chip_metadata = val_ds[ind]
                        z = task(x.unsqueeze(0))
                        out_path = join(pred_plot_dir, f'pred-{ind}.jpg')
                        if isinstance(z, dict):
                            z['output'] = z['output'].squeeze()
                            z['month_weights'] = z['month_weights'].squeeze()
                            z['month_pixel_weights'] = z['month_pixel_weights'].squeeze()
                            z['month_outputs'] = z['month_outputs'].squeeze()
                        BiomassDataset.plot_sample(
                            x, y.squeeze(), chip_metadata, z=z,
                            out_path=out_path)
                        out_paths.append(out_path)
            if wandb_logger is not None:
                wandb_logger.log_image(key="prediction-plots", images=out_paths)

        trainer.predict(task, datamodule=datamodule, return_predictions=False)

        with zipfile.ZipFile(pred_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, file_names in os.walk(pred_dir):
                for fn in file_names:
                    zipf.write(join(root, fn), arcname=fn)
        shutil.rmtree(pred_dir)

    if conf.program.s3_output_uri:
        subprocess.run(
            ['aws', 's3', 'sync', output_dir,
             join(conf.program.s3_output_uri, conf.program.run_name)])


if __name__ == '__main__':
    conf = set_up_omegaconf()
    pl.seed_everything(conf.program.seed)
    main(conf)
