import os
# Avoid Jax stealing all GPU memory.
from utils.utils import pprint_dict

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

import hydra
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from emlp.reps import Vector
from hydra.utils import get_original_cwd

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from nn.KFoldValidation import KFoldDataModule, KFoldValidation
from nn.LightningModel import LightningModel
from nn.EquivariantModules import EMLP, MLP
from datasets.com_momentum import COMMomentum

import optuna

from utils.robot_utils import get_robot_params
import logging

log = logging.getLogger(__name__)
cache_dir = None


def run_hp_search(cfg, network, dataset, trainer_kwargs, n_trials=100):
    study = optuna.create_study(direction="minimize", study_name=f"{cfg.model_type}_{cfg.robot_name}")
    state_dict = network.state_dict()
    study.optimize(lambda x: objective(x, cfg, network, dataset,
                                       state_dict=state_dict, trainer_kwargs=trainer_kwargs), n_trials=n_trials)
    results_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    results_df.sort_values("params_lr", inplace=True)
    results_df.to_csv("Results.csv")
    fig, ax =plt.subplots()
    ax.plot(results_df["params_lr"], results_df["value"], "-o")
    ax.set_ylabel("validation loss")
    ax.set_xlabel("lr")
    ax.set_xscale("log")
    ax.set_title(cfg.model_type)
    fig.savefig("lr-vs-validation_loss.png", dpi=90)
    fig.show()
    print(results_df)
    return results_df


def objective(trial: optuna.trial.Trial, cfg: DictConfig, network, dataset, trainer_kwargs, state_dict=None) -> float:

    lr = trial.suggest_loguniform(name="lr", low=1e-8, high=1e-1)
    # trial.
    # Build Lightning Module ___________________________________________________________________
    if state_dict is not None:  # Reinitialize weights
        network.load_state_dict(state_dict)
    # network.reset_parameters()
    pl_model = LightningModel(lr=lr, loss_fn=dataset.loss_fn,
                              metrics_fn=lambda x, y: dataset.compute_metrics(x, y, dataset.standarizer))
    pl_model.set_model(model=network)


    tb_logger = pl_loggers.TensorBoardLogger(".", name=f'trial_{trial.number}', version=0)
    ckpt_call = ModelCheckpoint(dirpath=tb_logger.log_dir, filename='best', monitor="hp/val_loss", save_last=True)
    stop_call = EarlyStopping(monitor="hp/val_loss", patience=100, mode='min')
    trainer_kwargs["callbacks"] = [ckpt_call, stop_call]
    trainer_kwargs['enable_progress_bar'] = False

    trainer = Trainer(logger=tb_logger, **trainer_kwargs)

    train_size = int((cfg.dataset.train_samples) * .8)
    val_size = int((cfg.dataset.train_samples) * .2)

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, cfg.dataset.test_samples, val_size],
                                                            generator=torch.Generator().manual_seed(cfg.seed))

    train_dataloader, val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=4,
                                                  collate_fn=lambda x: dataset.collate_fn(x)), \
                                       DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=4,
                                                  collate_fn=lambda x: dataset.collate_fn(x))

    ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
    best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)

    if best_path.exists():
        best_ckpt = torch.load(best_path)
        lr = best_ckpt['hyper_parameters']['lr']
        trial.params['lr'] = lr
        pl_model.lr = trial.params['lr']

    score = np.inf
    if best_path.exists() and not ckpt_path.exists():   # Experiment already finished
        for k, v in best_ckpt['callbacks'].items():
            if "EarlyStop" in k:
                score = v["best_score"].item()
    else:
        trainer.fit(pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                    ckpt_path=best_path if ckpt_path.exists() else None)
        score = trainer.checkpoint_callback.best_model_score.item()

        if isinstance(network, EMLP):
            log.info("Evaluating trained model equivariance")
            EMLP.test_module_equivariance(module=network, rep_in=network.rep_in, rep_out=network.rep_out)

    # Return best model validation score.
    return score


def get_network(model_type, Gin, Gout, num_layers, init_mode, ch, bias, cache_dir):
    if model_type == "emlp":
        network = EMLP(rep_in=Vector(Gin), rep_out=Vector(Gout), hidden_group=Gout, num_layers=num_layers,
                       ch=ch, init_mode=init_mode, activation=torch.nn.ReLU,
                       with_bias=bias, cache_dir=cache_dir).to(dtype=torch.float32)
    elif 'mlp' in model_type:
        if 'mean' in init_mode.lower(): return
        network = MLP(d_in=Gin.d, d_out=Gout.d, num_layers=num_layers, init_mode=init_mode,
                      ch=ch, with_bias=bias, activation=torch.nn.ReLU).to(dtype=torch.float32)
    else:
        raise NotImplementedError(model_type)
    log.info(network)
    return network


@hydra.main(config_path='cfg/supervised', config_name='config')
def main(cfg: DictConfig):
    log.info(f"XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")
    torch.set_default_dtype(torch.float32)
    cfg.seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    seed_everything(seed=np.random.randint(0, 1000))
    log.info(f"Current working directory : {os.getcwd()}")
    # Avoid repeating to compute basis at each experiment.
    root_path = pathlib.Path(get_original_cwd()).resolve()
    global cache_dir
    cache_dir = root_path.joinpath(".empl_cache")
    cache_dir.mkdir(exist_ok=True)

    log.info(f"Cache Directory exists {cache_dir.exists()}: {cache_dir}")
    assert cache_dir.exists(), cache_dir.absolute()

    robot, Gin_data, Gout_data, Gin_model, Gout_model, = get_robot_params(cfg.robot_name)

    # Configure base trainer parameters ________________________________________________________
    callbacks = [ModelCheckpoint(monitor="hp/val_loss", mode='min', dirpath='.', filename="best",
                                 save_last=True, every_n_epochs=1),
                 EarlyStopping(monitor="hp/val_loss", patience=cfg.max_epochs * 0.1, mode='min')]


    if cfg.run_type == "hp_search":
        trainer_kwargs = {'gpus': 1 if torch.cuda.is_available() else 0,
                          'accelerator': "auto",
                          # 'strategy': "ddp",
                          'log_every_n_steps': max(int(cfg.dataset.train_samples // cfg.batch_size), 50),
                          'max_epochs': cfg.max_epochs,
                          'check_val_every_n_epoch': 1,
                          'benchmark': True,
                          'callbacks': callbacks}
        # Prepare model ____________________________________________________________________________
        model_type = cfg.model_type.lower()
        network = get_network(model_type=model_type, Gin=Gin_model, Gout=Gout_model, num_layers=cfg.num_layers,
                              init_mode=cfg.init_mode, ch=cfg.num_channels, bias=cfg.bias, cache_dir=cache_dir)
        dataset = COMMomentum(robot, Gin=Gin_data, Gout=Gout_data, size=cfg.dataset.train_samples + cfg.dataset.test_samples,
                              angular_momentum=cfg.dataset.angular_momentum, standarize=True,
                              augment="aug" in model_type, dtype=torch.float32)
        results = run_hp_search(cfg=cfg, network=network, dataset=dataset, trainer_kwargs=trainer_kwargs, n_trials=cfg.lr_trials)
        print(results)
        # TODO: Process results.
    elif "cross_val" in cfg.run_type:
        trainer_kwargs = {'gpus': 1 if torch.cuda.is_available() else 0,
                          'accelerator': "auto",
                          # 'strategy': "ddp",
                          # 'log_every_n_steps': int(cfg.dataset.train_samples // cfg.batch_size // 2),
                          # Every half epoch.
                          'max_epochs': cfg.max_epochs,
                          'check_val_every_n_epoch': 1,
                          'benchmark': True,
                          'callbacks': callbacks}

        # init_modes = ['fan_in', 'fan_out', 'normal0.1', 'normal1.0']
        init_modes = [cfg.init_mode] if isinstance(cfg.init_mode, str) else cfg.init_mode
        train_sizes = cfg.dataset.train_samples
        # model_types = ['emlp', 'mlp', 'mlp-aug']
        model_types = cfg.model_type
        hidden_layers = cfg.num_layers
        hd = hidden_layers
        results = {}

        test_dataset = COMMomentum(robot, Gin=Gin_data, Gout=Gout_data, size=cfg.dataset.test_samples,
                                   angular_momentum=cfg.dataset.angular_momentum, standarize=cfg.dataset.standarize,
                                   augment=True, dtype=torch.float32)

        metrics = []
        print(train_sizes)
        for training_size in train_sizes:
            # Prepare data _____________________________________________________________________________
            train_dataset = COMMomentum(robot, Gin=Gin_model, Gout=Gout_model, size=training_size,
                                        angular_momentum=cfg.dataset.angular_momentum, standarize=False,
                                        augment=cfg.dataset.augmentation, dtype=torch.float32)
            # Use test set Standarizer
            train_dataset.X, train_dataset.Y = test_dataset.standarizer.transform(train_dataset.X, train_dataset.Y)

            # Set number of epochs accordingly to training size
            new_max_epochs = int(max(cfg.dataset.test_samples, training_size) *
                                 cfg.max_epochs / training_size)
            trainer_kwargs['max_epochs'] = new_max_epochs
            trainer_kwargs['log_every_n_steps'] = max(int(training_size // cfg.batch_size), 50)
            for model_type in model_types:
                if 'aug' in model_type:
                    train_dataset.augment = True
                else:
                    train_dataset.augment = False
                network = get_network(model_type=model_type, Gin=Gin_model, Gout=Gout_model, num_layers=hd,
                                      ch=cfg.num_channels, bias=cfg.bias, init_mode='fan_in', cache_dir=cache_dir)
                # Build Lightning Module
                pl_model = LightningModel(lr=cfg.lr, loss_fn=train_dataset.loss_fn,
                                          metrics_fn=lambda x, y: train_dataset.compute_metrics(x, y, train_dataset.standarizer))
                pl_model.set_model(model=network)

                for init_mode in init_modes:
                    if 'mean' in init_mode and not 'emlp' in model_type: continue
                    run_name = {"model": model_type, "hidden_layers": hd, "init_mode": init_mode,
                                "training_samples": training_size}
                    metrics = sorted(list(run_name.keys()))
                    # Re initialize parameters
                    pl_model.model.reset_parameters(init_mode=init_mode)

                    datamodule = KFoldDataModule(train_dataset=train_dataset, test_dataset=test_dataset,
                                                 batch_size=cfg.batch_size)

                    kfold_val = KFoldValidation(model=pl_model, trainer_kwargs=trainer_kwargs, reinitialize=True,
                                                kfold_data_module=datamodule, num_folds=cfg.kfolds,
                                                run_name=pprint_dict(run_name))
                    # if cfg.auto_lr:
                    #     # Tune Hyperparameters
                    #     tmp_trainer = Trainer(auto_lr_find=True, logger=False, gpus=1, deterministic=False,
                    #                           detect_anomaly=True)
                    #     pl_model.model.reset_parameters()
                    #     tmp_trainer.tune(pl_model, train_dataloaders=datamodule.train_dataloader(),
                    #                      val_dataloaders=datamodule.test_dataloader())
                    #     lr_finder = tmp_trainer.tuner.lr_find(pl_model,
                    #                                           train_dataloader=datamodule.train_dataloader(),
                    #                                           val_dataloaders=datamodule.test_dataloader())
                    #                                           # datamodule=datamodule)
                    #     # Results can be found in
                    #     r = lr_finder.results
                    #     new_lr = lr_finder.suggestion()
                    #     # Plot with
                    #     fig = lr_finder.plot(suggest=True)
                    #     plt.title(f"{model_type}")
                    #     fig.show()
                    results[str(run_name)] = kfold_val.run()

        summaries = KFoldValidation.summarize_results(results)
        test_pds, train_pds, val_pds = [], [], []
        for run_name, summary in summaries.items():
            test_pds.append(pd.DataFrame(summary['test']).assign(**eval(run_name)))
            train_pds.append(pd.DataFrame(summary['train']).assign(**eval(run_name)))
            val_pds.append(pd.DataFrame(summary['val']).assign(**eval(run_name)))

        test_pd = pd.concat(test_pds, axis=0, join='inner')
        train_pd = pd.concat(train_pds, axis=0, join='inner')
        val_pd = pd.concat(val_pds, axis=0, join='inner')

        file_name = f"{model_types}-{init_modes}-hd{hidden_layers}.csv"
        test_pd.to_csv(f"TEST-{file_name}")
        train_pd.to_csv(f"TRAIN-{file_name}")
        val_pd.to_csv(f"VAL-{file_name}")

        # Plot.
        test_pd_long = pd.melt(test_pd, id_vars=metrics, var_name="metric")
        metric_names = np.unique(test_pd_long['metric']).tolist()
        g = sns.catplot(x='training_samples', y='value', col='metric', row='init_mode', hue='model',
                        col_order=reversed(metric_names), row_order=init_modes,
                        data=test_pd_long, kind="violin", inner="box", palette='PuBuGn',
                        scale='count', scale_hue=True, split=False, sharey=False, sharex=False)
        g.figure.savefig(file_name + ".png", dpi=90)
        g.figure.show()
        # print(summaries)

    elif cfg.run_type == "single_run":
        pass
    #
    # # Build Lightning Module
    # model = LightningModel(model=network, lr=cfg.lr, loss_fn=dataset.loss_fn,
    #                        metrics_fn=dataset.compute_metrics)>
    #
    # # Configure Logger _________________________________________________________________________
    # tb_logger = pl_loggers.TensorBoardLogger(".", name=f'seed={cfg.seed}', version=cfg.seed)
    # gpu_available = torch.cuda.is_available()
    # log.warning(f"CUDA GPU available {torch.cuda.get_device_name(0) if gpu_available else 'False'}")
    # # TODO: Remove for parallel training.
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    #
    # }
    #
    # if cfg.kfold:
    #     num_folds = 5
    #     datamodule = KFoldDataModule(dataset, test_samples=cfg.dataset.test_samples, batch_size=cfg.batch_size)
    #     kfold_val = KFoldValidation(model=model, trainer_kwargs=trainer_kwargs,
    #                                 kfold_data_module=datamodule, num_folds=num_folds,
    #                                 export_path="kfold")
    #     a = kfold_val.run()
    #     print(kfold_val.summary())
    #
    # else:
    #     trainer = Trainer(**trainer_kwargs)
    #
    #     train_dataset, val_dataset = random_split(dataset, [cfg.dataset.train_samples - cfg.dataset.test_samples,
    #                                                         cfg.dataset.test_samples])
    #     train_dataloader, val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size), \
    #                                        DataLoader(val_dataset, batch_size=cfg.batch_size)
    #
    #     trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
