import copy
import os

import torch
from torch.utils.data.sampler import WeightedRandomSampler

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
                                            use_class_imbalance_w=False,
                                            window_size=cfg.dataset.window_size, device=device,
                                            partition=cfg.dataset.data_folder)

        val_dataset = UmichContactDataset(data_name="val.npy",
                                          label_name="val_label.npy", train_ratio=cfg.dataset.train_ratio,
                                          augment=False, use_class_imbalance_w=False,
                                          window_size=cfg.dataset.window_size, device=device,
                                          partition=cfg.dataset.data_folder)
        test_dataset = UmichContactDataset(data_name="test.npy",
                                           label_name="test_label.npy", train_ratio=cfg.dataset.train_ratio,
                                           augment=False, use_class_imbalance_w=False,
                                           window_size=cfg.dataset.window_size, device=device,
                                           partition=cfg.dataset.data_folder,
                                           )
        sampler = None
        if cfg.dataset.balanced_classes:
            class_freqs = torch.clone(train_dataset.contact_state_freq)
            # As dataset is heavily unbalanced, set maximum sampling prob to uniform sampling from contact_states.
            class_freqs = torch.maximum(class_freqs,
                                        torch.ones_like(class_freqs) * (1 / train_dataset.n_contact_states))
            class_freqs = class_freqs / torch.linalg.norm(class_freqs)
            sample_weights = 1 - (class_freqs[train_dataset.label])
            # a = sample_weights.cpu().numpy()
            sampler = WeightedRandomSampler(sample_weights, num_samples=cfg.dataset.batch_size, replacement=False)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.dataset.batch_size,
                                      shuffle=True if sampler is None else None, sampler=sampler,
                                      num_workers=cfg.num_workers, collate_fn=lambda x: train_dataset.collate_fn(x))
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg.dataset.batch_size,
                                    collate_fn=lambda x: val_dataset.collate_fn(x), num_workers=cfg.num_workers)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg.dataset.batch_size,
                                     collate_fn=lambda x: val_dataset.collate_fn(x), num_workers=cfg.num_workers)

    elif cfg.dataset.name == "com_momentum":
        robot, Gin_data, Gout_data, Gin_model, Gout_model, = get_robot_params(cfg.robot_name)
        robot_name = cfg.robot_name.lower()
        robot_name = robot_name if '-' not in robot_name else robot_name.split('-')[0]
        data_path = root_path.joinpath(f"datasets/com_momentum/{robot_name}")
        # Training only sees the model symmetries
        train_dataset = COMMomentum(robot, Gin=Gin_model, Gout=Gout_model, type='train', samples=cfg.dataset.samples,
                                    train_ratio=cfg.dataset.train_ratio, angular_momentum=cfg.dataset.angular_momentum,
                                    standarizer=cfg.dataset.standarize, augment=cfg.dataset.augment,
                                    data_path=data_path, dtype=torch.float32, device=device)
        # Test and validation use theoretical symmetry group, and training set standarization
        test_dataset = COMMomentum(robot, Gin=Gin_data, Gout=Gout_data, type='test', samples=cfg.dataset.samples,
                                   train_ratio=cfg.dataset.train_ratio, angular_momentum=cfg.dataset.angular_momentum,
                                   data_path=data_path,
                                   augment='hard', dtype=torch.float32, device=device,
                                   standarizer=train_dataset.standarizer)
        val_dataset = COMMomentum(robot, Gin=Gin_data, Gout=Gout_data, type='val', samples=cfg.dataset.samples,
                                  train_ratio=cfg.dataset.train_ratio, angular_momentum=cfg.dataset.angular_momentum,
                                  data_path=data_path, augment=True, dtype=torch.float32, device=device,
                                  standarizer=train_dataset.standarizer)

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


def fine_tune_model(cfg, best_ckpt_path: pathlib.Path, pl_model, batches_per_original_epoch, epochs,
                    test_dataloader, train_dataloader, val_dataloader, device, version='finetuned=True'):
    if not cfg.model.model_type.lower() in ['emlp', 'ecnn']: return

    assert best_ckpt_path.exists(), "Best model not found for finetunning"

    best_ckpt = torch.load(best_ckpt_path)
    pl_model.load_state_dict(best_ckpt['state_dict'])  # Load best weights

    # Freeze most of model model.
    pl_model.model.unfreeze_equivariance(num_layers=cfg.model.fine_tune_num_layers)

    # for name, parameter in pl_model.model.named_parameters():
    #     print(f"{parameter.requires_grad}: {name}")

    # Reduce the magnitude of the lr
    pl_model.lr *= cfg.model.fine_tune_lr_scale

    fine_tb_logger = pl_loggers.TensorBoardLogger(".", name=f'seed={cfg.seed}',
                                                  version=version,
                                                  default_hp_metric=False)
    ckpt_path = get_ckpt_storage_path(fine_tb_logger.log_dir, use_volatile=cfg.use_volatile)
    fine_ckpt_call = ModelCheckpoint(dirpath=ckpt_path, filename='best', monitor="val_loss",
                                     save_last=True)
    fine_stop_call = EarlyStopping(monitor='val_loss', patience=max(10, int(epochs * 0.1)), mode='min')
    exp_terminated, ckpt_path, best_ckpt_path = check_if_resume_experiment(fine_ckpt_call)

    fine_trainer = Trainer(gpus=1 if torch.cuda.is_available() and device != 'cpu' else 0,
                           logger=fine_tb_logger,
                           accelerator="auto",
                           log_every_n_steps=max(int(batches_per_original_epoch * cfg.dataset.log_every_n_epochs * 0.5),
                                                 50),
                           max_epochs=epochs * 1.5 if not cfg.debug_loops else 3,
                           check_val_every_n_epoch=1,
                           benchmark=True,
                           callbacks=[fine_ckpt_call, fine_stop_call],
                           fast_dev_run=cfg.debug,
                           detect_anomaly=cfg.debug,
                           resume_from_checkpoint=ckpt_path if ckpt_path.exists() else None,
                           limit_train_batches=1.0 if not cfg.debug_loops else 0.005,
                           limit_test_batches=1.0 if not cfg.debug_loops else 0.005,
                           limit_val_batches=1.0 if not cfg.debug_loops else 0.005,
                           )
    log_path = pathlib.Path(fine_tb_logger.log_dir)
    # test_model(path=log_path, trainer=fine_trainer, model=pl_model,
    #            train_dataloader=train_dataloader, test_dataloader=test_dataloader, val_dataloader=val_dataloader)


# def test_model(path, trainer, model, train_dataloader, test_dataloader, val_dataloader):
#     test_metrics = trainer.test(model=model, dataloaders=test_dataloader)[0]
#     df = pd.DataFrame.from_dict({k: [v] for k, v in test_metrics.items()})
#     path.mkdir(exist_ok=True, parents=True)
#     # noinspection PyTypeChecker
#     df.to_csv(str(path.joinpath("test_metrics.csv").absolute()))

def get_ckpt_storage_path(log_path, use_volatile=True):
    if not use_volatile: return log_path
    try:
        exp_path = pathlib.Path(*pathlib.Path(os.getcwd()).parts[3:])
        asi_root_folder = pathlib.Path(os.environ['NVME1DIR'])
        ckpt_path = asi_root_folder / exp_path / log_path
        ckpt_path.mkdir(exist_ok=True, parents=True)
        log.info(f"Using volatile storage {asi_root_folder} for checkpointing")
        return str(ckpt_path)
    except KeyError as e:
        log.warning(f"Volatile storage {e} not found, default to current log dir for model checkpointing: "
                    f"{pathlib.Path(log_path).resolve()}")
        return log_path


@hydra.main(config_path='cfg/supervised', config_name='config')
def main(cfg: DictConfig):
    log.info("\n\n NEW RUN \n\n")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    cfg.seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    cfg['debug'] = cfg.get('debug', False)
    cfg['debug_loops'] = cfg.get('debug_loops', False)
    seed_everything(seed=cfg.seed)

    root_path = pathlib.Path(get_original_cwd()).resolve()
    cache_dir = root_path.joinpath(".empl_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_dir = None  # if cfg.dataset.name == "com_momentum" else cache_dir

    # Check if experiment already run
    tb_logger = pl_loggers.TensorBoardLogger(".", name=f'seed={cfg.seed}', version=cfg.seed, default_hp_metric=False)
    ckpt_folder_path = get_ckpt_storage_path(tb_logger.log_dir, use_volatile=cfg.use_volatile)
    ckpt_call = ModelCheckpoint(dirpath=ckpt_folder_path, filename='best', monitor="val_loss", save_last=True)
    training_done, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)
    test_metrics_path = pathlib.Path(tb_logger.log_dir) / 'test_metrics.csv'
    training_done = True if test_metrics_path.exists() else training_done

    # Check if fine tunning is desired and if it has already run
    should_fine_tune = cfg.model.model_type.lower() in ['ecnn']

    finetune_folder_name = f'finetuned=True flrs={cfg.model.fine_tune_lr_scale} fly={cfg.model.fine_tune_num_layers}'
    if should_fine_tune:
        finetune_folder_path = ckpt_path.parent.parent / finetune_folder_name
        finetuned_ckpt_path, finetuned_best_path = (finetune_folder_path / ckpt_path.name, finetune_folder_path /
                                                    best_path.name)
        finetune_done = finetuned_best_path.exists() and not finetuned_ckpt_path.exists()
    else:
        finetune_done = True

    ## TODO: Avoid finetune for now
    finetune_done = True

    if not training_done or not finetune_done:
        # Prepare data
        datasets, dataloaders = get_datasets(cfg, device, root_path)
        train_dataset, val_dataset, test_dataset = datasets
        train_dataloader, val_dataloader, test_dataloader = dataloaders

        # Prepare model
        model = get_model(cfg.model, Gin=train_dataset.Gin, Gout=train_dataset.Gout, cache_dir=cache_dir)
        log.info(model)

        # Prepare Lightning
        test_set_metrics_fn = (lambda x: test_dataset.test_metrics(*x)) if hasattr(test_dataset,
                                                                                   'test_metrics') else None
        val_set_metrics_fn = (lambda x: val_dataset.test_metrics(*x)) if hasattr(val_dataset, 'test_metrics') else None
        pl_model = LightningModel(lr=cfg.model.lr, loss_fn=train_dataset.loss_fn,
                                  metrics_fn=lambda x, y: train_dataset.compute_metrics(x, y),
                                  test_epoch_metrics_fn=test_set_metrics_fn,
                                  val_epoch_metrics_fn=val_set_metrics_fn,
                                  )
        pl_model.set_model(model)

        original_dataset_samples = int(0.7 * len(train_dataset) / cfg.dataset.train_ratio)
        batches_per_original_epoch = original_dataset_samples // cfg.dataset.batch_size
        epochs = cfg.dataset.max_epochs * batches_per_original_epoch // (len(train_dataset) // cfg.dataset.batch_size)

        if not training_done:
            stop_call = EarlyStopping(monitor='val_loss', patience=max(10, int(epochs * 0.2)), mode='min')

            log.info("\n\nInitiating Training\n\n")
            trainer = Trainer(gpus=1 if torch.cuda.is_available() and device != 'cpu' else 0,
                              logger=tb_logger,
                              accelerator="auto",
                              log_every_n_steps=max(int(batches_per_original_epoch * cfg.dataset.log_every_n_epochs),
                                                    50),
                              max_epochs=epochs if not cfg.debug_loops else 3,
                              check_val_every_n_epoch=1,
                              # benchmark=True,
                              callbacks=[ckpt_call, stop_call],
                              fast_dev_run=cfg.debug,
                              # detect_anomaly=cfg.debug,
                              enable_progress_bar=True,
                              limit_train_batches=1.0 if not cfg.debug_loops else 0.005,
                              # limit_test_batches=1.0 if not cfg.debug_loops else 0.005,
                              limit_val_batches=1.0 if not cfg.debug_loops else 0.005,
                              resume_from_checkpoint=ckpt_path if ckpt_path.exists() else None,
                              )

            trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

            # Test model
            log.info("\n\nInitiating Testing\n\n")
            # log_path = pathlib.Path(tb_logger.log_dir)
            # test_model(path=log_path, trainer=trainer, model=pl_model,
            #            train_dataloader=train_dataloader, test_dataloader=test_dataloader,
            #            val_dataloader=val_dataloader)

        if not finetune_done:
            log.info("\n\nInitiating Fine-tuning\n\n")
            train_dataset.augment = True  # Fine tune with augmentation.
            fine_tune_model(cfg, best_path, pl_model, batches_per_original_epoch, epochs,
                            test_dataloader=test_dataloader, train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader, device=device, version=finetune_folder_name)
    else:
        log.warning(f"Experiment: {os.getcwd()} Already Finished. Ignoring run")


if __name__ == '__main__':
    main()
