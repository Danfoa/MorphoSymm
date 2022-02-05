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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)

    torch.set_default_dtype(torch.float32)
    cfg.seed = cfg.seed if cfg.seed > 0 else np.random.randint(0, 1000)
    seed_everything(seed=np.random.randint(0, 1000))
    print(f"Current working directory : {os.getcwd()}")
    # Avoid repeating to compute basis at each experiment.
    root_path = pathlib.Path(get_original_cwd()).resolve()
    cache_dir = root_path.joinpath(".empl_cache")
    cache_dir.mkdir(exist_ok=True)

    robot, G_in, GC = get_robot_params(cfg.robot_name)

    # Define output group for linear momentum
    G_out = GC.canonical_group(3)

    model_type = cfg.model_type.lower()
    if model_type == "emlp":
        network = EMLP(rep_in=Vector(G_in), rep_out=Vector(G_out),
                       group=C2, num_layers=cfg.num_layers, ch=cfg.num_channels,
                       with_bias=True, cache_dir=cache_dir).to(dtype=torch.float32)
    elif model_type == 'mlp':
        network = MLP(d_in=G_in.d, d_out=G_out.d, num_layers=cfg.num_layers,
                      ch=cfg.num_channels).to(dtype=torch.float32)
    else:
        raise NotImplementedError(model_type)

    model = EquivariantModel(model=network, model_type=model_type, lr=1e-2)

    dataset = COMMomentum(robot, cfg.dataset.num_samples, angular_momentum=cfg.dataset.angular_momentum)
    val_dataset = COMMomentum(robot, int(cfg.dataset.val_samples), angular_momentum=cfg.dataset.angular_momentum)
    train_dataloader, val_dataloader = DataLoader(dataset, batch_size=cfg.batch_size), \
                                       DataLoader(val_dataset, batch_size=cfg.batch_size)

    # Configure Logger
    tb_logger = pl_loggers.TensorBoardLogger(".", name=f'seed={cfg.seed}', version=cfg.seed)

    trainer = Trainer(logger=tb_logger, track_grad_norm=2, max_epochs=cfg.max_epochs, callbacks=[DeviceStatsMonitor()])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
