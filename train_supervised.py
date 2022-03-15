import os
import pathlib
import seaborn as sns

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from emlp.reps import Vector
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from nn.KFoldValidation import KFoldDataModule, KFoldValidation
from nn.LightningModel import LightningModel
from nn.EquivariantModules import EMLP, MLP
from nn.datasets import COMMomentum

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from utils.robot_utils import get_robot_params
import logging

log = logging.getLogger(__name__)
cache_dir = None


def run_hp_search(cfg, network, dataset, trainer_kwargs):
    study = optuna.create_study(direction="minimize", study_name=f"{cfg.model_type}_{cfg.robot_name}")
    state_dict = network.state_dict()
    study.optimize(lambda x: objective(x, cfg, network, dataset,
                                       state_dict=state_dict, trainer_kwargs=trainer_kwargs), n_trials=15)
    results_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    results_df.to_csv("Results.csv")
    print(results_df)
    return results_df


def objective(trial: optuna.trial.Trial, cfg: DictConfig, network, dataset, trainer_kwargs, state_dict=None) -> float:

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2)

    # Build Lightning Module ___________________________________________________________________
    if state_dict is not None:  # Reinitialize weights
        network.load_state_dict(state_dict)
    model = LightningModel(model=network, lr=lr, loss_fn=dataset.loss_fn, metrics_fn=dataset.compute_metrics)


    tb_logger = pl_loggers.TensorBoardLogger(".", name=f'trial_{trial.number}_lr={lr:.5f}', version=cfg.seed)
    trainer = Trainer(logger=tb_logger, **trainer_kwargs)

    train_size = int((cfg.dataset.train_samples) * .8)
    val_size = int((cfg.dataset.train_samples) * .2)

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, cfg.dataset.test_samples, val_size],
                                                            generator=torch.Generator().manual_seed(cfg.seed))

    train_dataloader, val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size), \
                                       DataLoader(val_dataset, batch_size=cfg.batch_size)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    if isinstance(network, EMLP):
        log.info("Evaluating trained model equivariance")
        EMLP.test_module_equivariance(module=network, rep_in=network.rep_in, rep_out=network.rep_out)

    return trainer.callback_metrics["hp/val_loss"].item()


def get_network(model_type, Gin, Gout, num_layers, init_mode, ch, bias):
    if model_type == "emlp":
        network = EMLP(rep_in=Vector(Gin), rep_out=Vector(Gout), hidden_group=Gout, num_layers=num_layers,
                       ch=ch, init_mode=init_mode, activation=torch.nn.ReLU,
                       with_bias=bias, cache_dir=cache_dir).to(dtype=torch.float32)
    elif model_type == 'mlp':
        if 'mean' in init_mode.lower(): return
        network = MLP(d_in=Gin.d, d_out=Gout.d, num_layers=num_layers, init_mode=init_mode,
                      ch=ch, with_bias=bias, activation=torch.nn.ReLU).to(dtype=torch.float32)
    else:
        raise NotImplementedError(model_type)
    return network


@hydra.main(config_path='cfg/supervised', config_name='config')
def main(cfg: DictConfig):
    print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")
    torch.set_default_dtype(torch.float32)
    cfg.seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    seed_everything(seed=np.random.randint(0, 1000))
    print(f"Current working directory : {os.getcwd()}")

    # Avoid repeating to compute basis at each experiment.
    root_path = pathlib.Path(get_original_cwd()).resolve()
    cache_dir = root_path.joinpath(".empl_cache")
    cache_dir.mkdir(exist_ok=True)

    robot, Gin, Gout, GC = get_robot_params(cfg.robot_name)


    # Prepare data _____________________________________________________________________________
    dataset = COMMomentum(robot, cfg.dataset.train_samples + cfg.dataset.test_samples,
                          angular_momentum=cfg.dataset.angular_momentum, normalize=cfg.dataset.normalize,
                          augment=cfg.dataset.augmentation, Gin=Gin, Gout=Gout, dtype=np.float32)

    # Configure base trainer parameters ________________________________________________________
    chkp_callback = ModelCheckpoint(monitor="hp/val_loss", mode='min', dirpath='.', #f'trial_{trial.number}_lr={lr:.5f}',
                                    filename=f"best_model")
    early_stop_callback = EarlyStopping(monitor="hp/val_loss", patience=int(cfg.max_epochs * 0.5), mode='min')
    callbacks = [chkp_callback, early_stop_callback]

    trainer_kwargs = {'gpus': 1 if torch.cuda.is_available() else 0,
                      'accelerator': "auto",
                      'strategy': "ddp",
                      'log_every_n_steps': max(int(cfg.dataset.test_samples // cfg.batch_size // 2), 10),
                      'max_epochs': cfg.max_epochs,
                      'check_val_every_n_epoch': 1,
                      'benchmark': True,
                      'callbacks': callbacks}


    if cfg.run_type == "hp_search":
        # Prepare model ____________________________________________________________________________
        model_type = cfg.model_type.lower()
        network = get_network(model_type=model_type, Gin=Gin, Gout=Gout, num_layers=cfg.num_layers,
                              init_mode=cfg.int_mode, ch=cfg.num_channels, bias=cfg.with_bias)
        results = run_hp_search(cfg, model_type, network, trainer_kwargs)

        # TODO: Process results.
    elif "cross_val" in cfg.run_type:
        init_modes = ['fan_in', 'fan_out', 'normal0.1', 'normal1.0']
        results = {}
        model_types = ['emlp', 'mlp']
        for model_type in model_types:
            for init_mode in init_modes:
                run_name = model_type + "-" + init_mode
                network = get_network(model_type=model_type, Gin=Gin, Gout=Gout, num_layers=cfg.num_layers,
                                      init_mode=init_mode, ch=cfg.num_channels, bias=cfg.bias)

                # Build Lightning Module
                model = LightningModel(model=network, lr=cfg.lr, loss_fn=dataset.loss_fn,
                                       metrics_fn=dataset.compute_metrics)

                datamodule = KFoldDataModule(dataset, test_samples=cfg.dataset.test_samples, batch_size=cfg.batch_size)
                kfold_val = KFoldValidation(model=model, trainer_kwargs=trainer_kwargs, reinitialize=True,
                                            kfold_data_module=datamodule, num_folds=cfg.kfolds, export_path="kfold",
                                            run_name=run_name)

                results[run_name] = kfold_val.run()
        summaries = KFoldValidation.summarize_results(results)

        test_pd = pd.concat([pd.DataFrame(v['test']).assign(init=k.split('-')[1], model=k.split('-')[0]) for k, v in summaries.items()], axis=0, join='inner')
        test_pd_long = pd.melt(test_pd, id_vars=["init", "model"], var_name="metric")
        g = sns.catplot(x='model', y='value', data=test_pd_long, kind="violin",
                        inner="box", pallete=sns.color_palette("mako", as_cmap=True), scale='count',
                        scale_hue=True, split=False, col='metric', row='init', sharey=False)
        g.figure.show()
        g.figure.savefig("-".join(model_types) + "--" + "-".join(init_modes)+".png", dpi=90)
        g.figure.show()
        print(summaries)

    elif cfg.run_type == "single_run":
        pass

    print("5")
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
