import os
import pathlib

import hydra
import numpy as np
import torch
from emlp.reps import Vector
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import DeviceStatsMonitor
from torch.utils.data import DataLoader

from groups.SymmetricGroups import C2
from nn.EquivariantModel import EquivariantModel
from nn.EquivariantModules import EMLP, MLP
from nn.datasets import COMMomentum
from robots.bolt.BoltBullet import BoltBullet
import utils.utils

import logging
log = logging.getLogger(__name__)

def get_robot_params(robot_name):
    if robot_name.lower() == 'bolt':
        robot = BoltBullet()
        pq = [3, 4, 5, 0, 1, 2]
        pq.extend((np.array(pq) + len(pq)).tolist())
        rq = [-1, 1, 1, -1, 1, 1]
        rq.extend(rq)
        h_in = C2.oneline2matrix(oneline_notation=pq, reflexions=rq)
        Gin = C2(h_in)
        h_out = C2.oneline2matrix(oneline_notation=[0, 1, 2], reflexions=[1, -1, 1])
        Gout = C2(h_out)
    else:
        # robot_solo = Solo12Bullet()
        # pq = [(6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)]
        # rq = [(-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
        # H_solo = list(Sym3.oneline2matrix(oneline_notation=p, reflexions=r) for p, r in zip(pq, rq))
        # G_solo = Sym3(H_solo)
        # G = G_solo
        # robot = robot_solo
        raise NotImplementedError()

    return robot, Gin, Gout, C2


@hydra.main(config_path='cfg/supervised', config_name='config')
def main(cfg: DictConfig):
    print(f"XLA_PYTHON_CLIENT_PREALLOCATE: {os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")
    torch.set_default_dtype(torch.float32)
    cfg.seed = cfg.seed if cfg.seed > 0 else np.random.randint(0, 1000)
    seed_everything(seed=np.random.randint(0, 1000))
    print(f"Current working directory : {os.getcwd()}")

    # Avoid repeating to compute basis at each experiment.
    root_path = pathlib.Path(get_original_cwd()).resolve()
    cache_dir = root_path.joinpath(".empl_cache")
    cache_dir.mkdir(exist_ok=True)

    robot, Gin, Gout, GC = get_robot_params(cfg.robot_name)

    # Prepare model ____________________________________________________________________________
    model_type = cfg.model_type.lower()
    if model_type == "emlp":
        network = EMLP(rep_in=Vector(Gin), rep_out=Vector(Gout), group=C2, num_layers=cfg.num_layers,
                       ch=cfg.num_channels, init_mode=cfg.init_mode, activation=torch.nn.ReLU,
                       with_bias=cfg.bias, cache_dir=cache_dir).to(dtype=torch.float32)
    elif model_type == 'mlp':
        if 'mean' in cfg.init_mode.lower(): return
        network = MLP(d_in=Gin.d, d_out=Gout.d, num_layers=cfg.num_layers, init_mode=cfg.init_mode,
                      ch=cfg.num_channels, with_bias=cfg.bias, activation=torch.nn.ReLU).to(dtype=torch.float32)
    else:
        raise NotImplementedError(model_type)

    log.info(f"Network:\n{network}")
    model = EquivariantModel(model=network, model_type=model_type, lr=cfg.lr)

    # Prepare data _____________________________________________________________________________
    dataset = COMMomentum(robot, cfg.dataset.num_samples, angular_momentum=cfg.dataset.angular_momentum,
                          augment=cfg.with_augmentation, Gin=Gin, Gout=Gout)
    val_dataset = COMMomentum(robot, int(cfg.dataset.val_samples), angular_momentum=cfg.dataset.angular_momentum,
                              augment=False, Gin=Gin, Gout=Gout)
    train_dataloader, val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size), \
                                       DataLoader(val_dataset, batch_size=cfg.batch_size)

    # Configure Logger _________________________________________________________________________
    tb_logger = pl_loggers.TensorBoardLogger(".", name=f'seed={cfg.seed}', version=cfg.seed)
    # Configure GPU
    gpu_available = torch.cuda.is_available()
    log.warning(f"CUDA GPU available {torch.cuda.get_device_name(0) if gpu_available else 'False'}")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    trainer = Trainer(logger=tb_logger,
                      gpus=1 if gpu_available else 0,
                      # auto_select_gpus=True,
                      track_grad_norm=2,
                      log_every_n_steps=cfg.batch_size*10,
                      max_epochs=cfg.max_epochs,
                      callbacks=[DeviceStatsMonitor()],
                      check_val_every_n_epoch=1,
                      benchmark=True)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
