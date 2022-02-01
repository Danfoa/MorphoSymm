import os
from random import random

import numpy as np
import torch

from emlp.reps import V, Vector
from emlp.groups import O
from emlp.reps.linear_operators import densify
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from groups.SymmetricGroups import Sym3, C2
from nn.EquivariantModel import EquivariantModel
from robots.PinBulletWrapper import PinBulletWrapper
from robots.bolt.BoltBullet import BoltBullet
from robot_kinematic_symmetries import JointSpaceSymmetry, SemiDirectProduct
from nn.datasets import COMMomentum
from robots.solo.Solo12Bullet import Solo12Bullet
from utils.emlp_visualization import visualize_basis_ind

from pytorch_lightning import loggers as pl_loggers

from nn.EquivariantModules import BasisLinear, EMLP, MLP

from omegaconf import DictConfig, OmegaConf
import hydra


def get_robot_params(robot_name):
    if robot_name.lower() == 'bolt':
        robot = BoltBullet()
        pq = [3, 4, 5, 0, 1, 2]
        pq.extend((np.array(pq) + len(pq)).tolist())
        rq = [-1, 1, 1, -1, 1, 1]
        rq.extend(rq)
        h = C2.oneline2matrix(oneline_notation=pq, reflexions=rq)
        G = C2(h)
    else:
        # robot_solo = Solo12Bullet()
        # pq = [(6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)]
        # rq = [(-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
        # H_solo = list(Sym3.oneline2matrix(oneline_notation=p, reflexions=r) for p, r in zip(pq, rq))
        # G_solo = Sym3(H_solo)
        # G = G_solo
        # robot = robot_solo
        raise NotImplementedError()

    return robot, G, C2


@hydra.main(config_path='cfg/supervised', config_name='config')
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)
    cfg.seed = cfg.seed if cfg.seed > 0 else np.random.randint(0, 1000)
    seed_everything(seed=np.random.randint(0, 1000))
    print(f"Current working directory : {os.getcwd()}")

    robot, G_in, GC = get_robot_params(cfg.robot_name)

    # Define output group for linear momentum
    G_out = GC.canonical_group(3)

    model_type = cfg.model_type.lower()
    if model_type == "emlp":
        network = EMLP(rep_in=Vector(G_in), rep_out=Vector(G_out),
                       group=C2, num_layers=cfg.num_layers, ch=cfg.num_channels,
                       with_bias=True).to(dtype=torch.float32)
    elif model_type == 'mlp':
        network = MLP(d_in=G_in.d, d_out=G_out.d, num_layers=cfg.num_layers,
                      ch=cfg.num_channels).to(dtype=torch.float32)
    else:
        raise NotImplementedError(model_type)

    model = EquivariantModel(model=network, model_type=model_type, lr=1e-2)

    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    dataset = COMMomentum(robot, cfg.dataset.num_samples, angular_momentum=cfg.dataset.angular_momentum)
    val_dataset = COMMomentum(robot, int(cfg.dataset.num_samples * 0.1), angular_momentum=cfg.dataset.angular_momentum)
    train_dataloader, val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size), \
                                       DataLoader(val_dataset, batch_size=cfg.batch_size)

    # Configure Logger

    tb_logger = pl_loggers.TensorBoardLogger(".")
    trainer = Trainer(overfit_batches=1, logger=tb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
