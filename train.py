# Code in this file was adapted from
# https://github.com/microsoft/torchgeo/blob/main/train.py
import os
from os.path import join
from typing import Any, Dict, Tuple, Type, cast
import warnings
import zipfile
import shutil

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import s3fs

from biomass.models import TemporalPixelRegression, BiomassPredictionWriter
from biomass.dataset import BiomassDataModule


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
    experiment_name = conf.experiment.name
    if os.path.isfile(conf.program.output_dir):
        raise NotADirectoryError('`program.output_dir` must be a directory')
    os.makedirs(conf.program.output_dir, exist_ok=True)

    experiment_dir = join(conf.program.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    pred_dir = join(conf.program.output_dir, 'test-preds')
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = join(conf.program.output_dir, 'test-preds.zip')

    if len(os.listdir(experiment_dir)) > 0:
        if conf.program.overwrite:
            warnings.warn(
                f'WARNING! The experiment directory, {experiment_dir}, already exists, '
                'we might overwrite data in it!')
        else:
            raise FileExistsError(
                f"The experiment directory, {experiment_dir}, already exists and isn't "
                "empty. We don't want to overwrite any existing results, exiting..."
            )

    with open(join(experiment_dir, 'experiment_config.yaml'), 'w') as f:
        OmegaConf.save(config=conf, f=f)

    task_args = OmegaConf.to_object(conf.experiment.task)
    task = TemporalPixelRegression(**task_args)
    datamodule_args = OmegaConf.to_object(conf.experiment.datamodule)
    datamodule = BiomassDataModule(**datamodule_args)

    if conf.program.train:
        csv_logger = pl_loggers.CSVLogger(conf.program.log_dir, name=experiment_name)

        monitor_metric = 'val_loss'
        mode = 'min'

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric, dirpath=experiment_dir, save_top_k=1, save_last=True)
        early_stopping_callback = EarlyStopping(
            monitor=monitor_metric, min_delta=0.00, patience=18, mode=mode)
        pred_writer = BiomassPredictionWriter(
            output_dir=pred_dir, write_interval='batch')

        trainer_args = OmegaConf.to_object(conf.trainer)
        trainer_args['callbacks'] = [
            checkpoint_callback, early_stopping_callback, pred_writer]
        trainer_args['logger'] = csv_logger
        trainer_args['default_root_dir'] = experiment_dir

        trainer = pl.Trainer(**trainer_args)

        if trainer_args.get('auto_lr_find'):
            trainer.tune(model=task, datamodule=datamodule)

        trainer.fit(model=task, datamodule=datamodule)

    if conf.program.predict:
        ckpt_path = join(experiment_dir, 'last.ckpt')
        task = TemporalPixelRegression.load_from_checkpoint(ckpt_path)
        trainer.predict(task, datamodule=datamodule, return_predictions=False)

        with zipfile.ZipFile(pred_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, file_names in os.walk(pred_dir):
                for fn in file_names:
                    zipf.write(join(root, fn), arcname=fn)
        shutil.rmtree(pred_dir)

    s3 = s3fs.S3FileSystem()
    s3.put(conf.program.output_dir, conf.program.s3_output_dir, recursive=True)


if __name__ == '__main__':
    conf = set_up_omegaconf()
    pl.seed_everything(conf.program.seed)
    main(conf)
