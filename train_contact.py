import hydra
import os

from torch import optim

from utils.utils import check_if_resume_experiment

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from datasets.umich_contact_dataset import UmichContactDataset
from groups.SemiDirectProduct import SparseRep
from nn.LightningModel import LightningModel

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import pathlib
import deep_contact_estimator
from deep_contact_estimator.src.train import train
from deep_contact_estimator.src.contact_cnn import *
from deep_contact_estimator.utils.data_handler import *
from deep_contact_estimator.utils.mat2numpy import mat2numpy_split
from nn.ContactECNN import ContactECNN

import logging
log = logging.getLogger(__name__)

def get_model(cfg, Gin=None, Gout=None, cache_dir=None):
    if "ecnn" in cfg.model_type.lower():
        model = ContactECNN(SparseRep(Gin), SparseRep(Gout), Gin, cache_dir=cache_dir, dropout=cfg.dropout)
    elif "cnn" == cfg.model_type.lower():
        model = contact_cnn()
    else:
        raise NotImplementedError(cfg.model_type)
    return model

def ensure_training_partition(data_path, train_ratio:float = 0.7) -> pathlib.Path:

    mat_path = pathlib.Path(data_path).parent.joinpath('mat')
    run_partition= pathlib.Path(str(data_path) + f'_train_ratio={train_ratio}')

    if not run_partition.joinpath('train.npy').exists():
        run_partition.mkdir(exist_ok=True)
        log.info(f"Generating dataset and saving it to {run_partition}")
        mat2numpy_split(data_pth=str(mat_path) + "/", save_pth=str(run_partition) + "/",
                        train_ratio=train_ratio, val_ratio=0.15)
        train_np = run_partition.joinpath('train.npy')
        assert train_np.exists(), train_np.absolute()
        return run_partition
    return pathlib.Path(data_path)

@hydra.main(config_path='cfg/supervised', config_name='config')
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    seed_everything(seed=np.random.randint(0, 1000))

    root_path = pathlib.Path(get_original_cwd()).resolve()
    cache_dir = root_path.joinpath(".empl_cache")
    cache_dir.mkdir(exist_ok=True)

    data_path = pathlib.Path(deep_contact_estimator.__file__).parents[1]
    cfg.dataset.data_folder = str(data_path.joinpath(cfg.dataset.data_folder))

    cfg.dataset = cfg.dataset
    print("Using the following params: ")
    print("-------------path-------------")
    print("data_folder: ", cfg.dataset.data_folder)

    data_folder = ensure_training_partition(cfg.dataset.data_folder, train_ratio=cfg.dataset.train_ratio)

    # load data
    train_dataset = UmichContactDataset(data_path=data_folder.joinpath("train.npy"),
                                        label_path=data_folder.joinpath("train_label.npy"),
                                        augment=cfg.dataset.augment, use_class_imbalance_w=cfg.dataset.balanced_classes,
                                        window_size=cfg.dataset.window_size, device=device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True,
                                  num_workers=0,
                                  collate_fn=lambda x: train_dataset.collate_fn(x))

    val_data = UmichContactDataset(data_path=data_folder.joinpath("val.npy"),
                                   label_path=data_folder.joinpath("val_label.npy"),
                                   augment=False, use_class_imbalance_w=cfg.dataset.balanced_classes,
                                   window_size=cfg.dataset.window_size, device=device)
    val_dataloader = DataLoader(dataset=val_data, batch_size=cfg.dataset.batch_size, shuffle=False,
                                collate_fn=lambda x: val_data.collate_fn(x), num_workers=0)

    model = get_model(cfg.model, Gin=train_dataset.Gin, Gout=train_dataset.Gout, cache_dir=cache_dir)
    model.to(device)

    pl_model = LightningModel(lr=cfg.model.lr, loss_fn=train_dataset.loss_fn,
                              metrics_fn=lambda x, y: train_dataset.compute_metrics(x, y))
    pl_model.set_model(model)

    tb_logger = pl_loggers.TensorBoardLogger(".", name=f'seed={cfg.seed}', version=cfg.seed)
    ckpt_call = ModelCheckpoint(dirpath=tb_logger.log_dir, filename='best', monitor="val_loss", save_last=True)
    exp_terminated, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)

    if not exp_terminated:
        trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                          logger=tb_logger,
                          accelerator="auto",
                          log_every_n_steps=max(int((len(train_dataset) // cfg.dataset.batch_size) * cfg.dataset.log_every_n_epochs), 50),
                          max_epochs=cfg.dataset.max_epochs,
                          check_val_every_n_epoch=1,
                          benchmark=True,
                          callbacks=[ckpt_call],
                          fast_dev_run=cfg.get('debug', False),
                          detect_anomaly=cfg.get('debug', False),
                          resume_from_checkpoint=ckpt_path if ckpt_path.exists() else None,
                          num_sanity_val_steps=0,  # Lightning Bug.
                          )

        trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    else:
        log.warning(f"Experiment Already Finished. Ignoring run")

    # optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr)
    # model.train()
    # for sample in train_dataloader:
    #     optimizer.zero_grad()
    #
    #     x, y = sample
    #     output = pl_model(x)
    #     loss = train_dataset.loss_fn(output, y)
    #
    #     loss.backward()
    #     optimizer.step()
    #     print("Done")
    #     break


    #
    # model.to(device)
    # trainer.fit(pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # import torch
    # import torchvision.models as models
    # from torch.profiler import profile, record_function, ProfilerActivity
    #
    # model = get_model('ecnn', Gin=train_dataset.Gin, Gout=train_dataset.Gout, cache_dir=cache_dir)
    # model.to(device)
    # print(model)
    # with profile(activities=[
    #     ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     with record_function("model_inference"):
    #         train(model, train_dataloader, val_dataloader, cfg.dataset)
    #
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # prof.export_chrome_trace(f"{pwd}/ecnn_trace.json")
    #
    # print("_______________________________________________________________________________CNN")
    # model = get_model('cnn', Gin=train_dataset.Gin, Gout=train_dataset.Gout, cache_dir=cache_dir)
    # model.to(device)
    # print(model)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     with record_function("model_inference"):
    #         train(model, train_dataloader, val_dataloader, cfg.dataset)
    #
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # prof.export_chrome_trace(f"{pwd}/cnn_trace.json")

    # train(model, train_dataloader, val_dataloader, cfg.dataset)


if __name__ == '__main__':
    main()


