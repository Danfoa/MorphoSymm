import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pandas as pd

from datasets.com_momentum.com_momentum import COMMomentum
from datasets.umich_contact_dataset import UmichContactDataset
from nn.EquivariantModules import MLP, EMLP
from utils.robot_utils import get_robot_params

import hydra
from utils.utils import check_if_resume_experiment

from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from groups.SemiDirectProduct import SparseRep
from nn.LightningModel import LightningModel

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import pathlib
from deep_contact_estimator.src.contact_cnn import *
from deep_contact_estimator.utils.data_handler import *
from nn.ContactECNN import ContactECNN

import logging

log = logging.getLogger(__name__)


def get_model(cfg, Gin=None, Gout=None, cache_dir=None):
    if "ecnn" in cfg.model_type.lower():
        model = ContactECNN(SparseRep(Gin), SparseRep(Gout), Gin, cache_dir=cache_dir, dropout=cfg.dropout,
                            init_mode=cfg.init_mode, inv_dim_scale=cfg.inv_dims_scale)
    elif "cnn" == cfg.model_type.lower():
        model = contact_cnn()
    elif "emlp" == cfg.model_type.lower():
        model = EMLP(rep_in=SparseRep(Gin), rep_out=SparseRep(Gout), hidden_group=Gout, num_layers=cfg.num_layers,
                     ch=cfg.num_channels, init_mode=cfg.init_mode, activation=torch.nn.ReLU,
                     with_bias=cfg.bias, cache_dir=cache_dir, inv_dims_scale=cfg.inv_dims_scale).to(dtype=torch.float32)
    elif 'mlp' == cfg.model_type.lower():
        model = MLP(d_in=Gin.d, d_out=Gout.d, num_layers=cfg.num_layers, init_mode=cfg.init_mode,
                    ch=cfg.num_channels, with_bias=cfg.bias, activation=torch.nn.ReLU).to(dtype=torch.float32)
    else:
        raise NotImplementedError(cfg.model_type)
    return model


def get_datasets(cfg, device, root_path):
    if cfg.dataset.name == "contact":
        train_dataset = UmichContactDataset(data_name="train.npy",
                                            label_name="train_label.npy", train_ratio=cfg.dataset.train_ratio,
                                            augment=cfg.dataset.augment,
                                            use_class_imbalance_w=cfg.dataset.balanced_classes,
                                            window_size=cfg.dataset.window_size, device=device)

        val_dataset = UmichContactDataset(data_name="val.npy",
                                          label_name="val_label.npy", train_ratio=cfg.dataset.train_ratio,
                                          augment=False, use_class_imbalance_w=False,
                                          window_size=cfg.dataset.window_size, device=device)
        test_dataset = UmichContactDataset(data_name="test.npy",
                                           label_name="test_label.npy", train_ratio=cfg.dataset.train_ratio,
                                           augment=False, use_class_imbalance_w=False,
                                           window_size=cfg.dataset.window_size, device=device)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True,
                                      num_workers=cfg.num_workers, collate_fn=lambda x: train_dataset.collate_fn(x))
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg.dataset.batch_size,
                                    collate_fn=lambda x: val_dataset.collate_fn(x), num_workers=cfg.num_workers)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg.dataset.batch_size,
                                     collate_fn=lambda x: val_dataset.collate_fn(x), num_workers=cfg.num_workers)

    elif cfg.dataset.name == "com_momentum":
        robot, Gin_data, Gout_data, Gin_model, Gout_model, = get_robot_params(cfg.robot_name)
        data_path = root_path.joinpath(f"datasets/com_momentum/{cfg.robot_name.lower()}")
        # Training only sees the model symmetries
        train_dataset = COMMomentum(robot, Gin=Gin_model, Gout=Gout_model, type='train',
                                    train_ratio=cfg.dataset.train_ratio, angular_momentum=cfg.dataset.angular_momentum,
                                    standarizer=cfg.dataset.standarize, augment=cfg.dataset.augmentation,
                                    data_path=data_path, dtype=torch.float32, device=device)
        # Test and validation use theoretical symmetry group, and training set standarization
        test_dataset = COMMomentum(robot, Gin=Gin_data, Gout=Gout_data, type='test', train_ratio=cfg.dataset.train_ratio,
                                   angular_momentum=cfg.dataset.angular_momentum, data_path=data_path,
                                   augment=True, dtype=torch.float32, device=device, standarizer=train_dataset.standarizer)
        val_dataset = COMMomentum(robot, Gin=Gin_data, Gout=Gout_data, type='val', train_ratio=cfg.dataset.train_ratio,
                                  angular_momentum=cfg.dataset.angular_momentum, data_path=data_path,
                                  augment=True, dtype=torch.float32, device=device, standarizer=train_dataset.standarizer)

        train_dataloader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, num_workers=cfg.num_workers,
                                      shuffle=True, collate_fn=lambda x: train_dataset.collate_fn(x))
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, num_workers=cfg.num_workers,
                                    collate_fn=lambda x: val_dataset.collate_fn(x))
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, num_workers=cfg.num_workers,
                                     collate_fn=lambda x: test_dataset.collate_fn(x))

    else:
        raise NotImplementedError(cfg.dataset.name)

    datasets = train_dataset, val_dataset, test_dataset
    dataloaders = train_dataloader, val_dataloader, test_dataloader
    return datasets, dataloaders


@hydra.main(config_path='cfg/supervised', config_name='config')
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    cfg.seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    seed_everything(seed=cfg.seed)

    root_path = pathlib.Path(get_original_cwd()).resolve()
    cache_dir = root_path.joinpath(".empl_cache")
    cache_dir.mkdir(exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(".", name=f'seed={cfg.seed}', version=cfg.seed, default_hp_metric=False)
    ckpt_call = ModelCheckpoint(dirpath=tb_logger.log_dir, filename='best', monitor="val_loss", save_last=True)
    stop_call = EarlyStopping(monitor='val_loss', patience=max(10, int(cfg.dataset.max_epochs * 0.33)), mode='min')
    exp_terminated, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)

    if not exp_terminated:
        # Prepare data
        datasets, dataloaders = get_datasets(cfg, device, root_path)
        train_dataset, val_dataset, test_dataset = datasets
        train_dataloader, val_dataloader, test_dataloader = dataloaders

        # Prepare model
        model = get_model(cfg.model, Gin=train_dataset.Gin, Gout=train_dataset.Gout, cache_dir=cache_dir)
        log.info(model)

        # Prepare Lightning
        pl_model = LightningModel(lr=cfg.model.lr, loss_fn=train_dataset.loss_fn,
                                  metrics_fn=lambda x, y: train_dataset.compute_metrics(x, y))
        pl_model.set_model(model)

        original_dataset_samples = int(0.7 * len(train_dataset) / cfg.dataset.train_ratio)
        batches_per_original_epoch = original_dataset_samples // cfg.dataset.batch_size
        epochs = cfg.dataset.max_epochs * batches_per_original_epoch / (len(train_dataset) // cfg.dataset.batch_size)

        trainer = Trainer(gpus=1 if torch.cuda.is_available() and device != 'cpu' else 0,
                          logger=tb_logger,
                          accelerator="auto",
                          log_every_n_steps=max(
                              int((len(train_dataset) // cfg.dataset.batch_size) * cfg.dataset.log_every_n_epochs), 50),
                          max_epochs=epochs,
                          check_val_every_n_epoch=1,
                          benchmark=True,
                          callbacks=[ckpt_call, stop_call],
                          fast_dev_run=cfg.get('debug', False),
                          detect_anomaly=cfg.get('debug', False),
                          resume_from_checkpoint=ckpt_path if ckpt_path.exists() else None,
                          # num_sanity_val_steps=1,  # Lightning Bug.
                          )

        trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        test_metrics = trainer.test(model=pl_model, dataloaders=test_dataloader)[0]
        df = pd.DataFrame.from_dict({k: [v] for k, v in test_metrics.items()})
        path = pathlib.Path(tb_logger.log_dir)
        path.mkdir(exist_ok=True, parents=True)
        df.to_csv(str(path.joinpath("test_metrics.csv").absolute()))

    else:
        log.warning(f"Experiment: {os.getcwd()} Already Finished. Ignoring run")


if __name__ == '__main__':
    main()
