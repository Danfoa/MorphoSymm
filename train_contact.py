import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from datasets.umich_contact_dataset import UmichContactDataset
from groups.SemiDirectProduct import SparseRep

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import pathlib
import deep_contact_estimator
from deep_contact_estimator.src.train import train
from deep_contact_estimator.src.contact_cnn import *
from deep_contact_estimator.utils.data_handler import *
from nn.ContactECNN import ContactECNN


def get_model(model_type, Gin=None, Gout=None, cache_dir=None):
    if "ecnn" in model_type.lower():
        model = ContactECNN(SparseRep(Gin), SparseRep(Gout), Gin, cache_dir=cache_dir)
    elif "cnn" == model_type.lower():
        model = contact_cnn()
    else:
        raise NotImplementedError(model_type)
    return model

@hydra.main(config_path='cfg/supervised', config_name='config')
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    root_path = pathlib.Path(get_original_cwd()).resolve()
    cache_dir = root_path.joinpath(".empl_cache")
    cache_dir.mkdir(exist_ok=True)

    data_path = pathlib.Path(deep_contact_estimator.__file__).parents[1]
    cfg.dataset.data_folder = str(data_path.joinpath(cfg.dataset.data_folder))

    cfg.dataset = cfg.dataset
    print("Using the following params: ")
    print("-------------path-------------")
    print("data_folder: ", cfg.dataset.data_folder)

    # load data
    train_dataset = UmichContactDataset(data_path=cfg.dataset.data_folder + "/train.npy",
                                     label_path=cfg.dataset.data_folder + "/train_label.npy",
                                     augment=cfg.dataset.augment,
                                     window_size=cfg.dataset.window_size, device=device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.dataset.batch_size,
                                  shuffle=cfg.dataset.shuffle,
                                  collate_fn=lambda x: train_dataset.collate_fn(x))
    val_data = UmichContactDataset(data_path=cfg.dataset.data_folder + "/val.npy",
                                   label_path=cfg.dataset.data_folder + "/val_label.npy",
                                   augment=cfg.dataset.augment,
                                   window_size=cfg.dataset.window_size, device=device)
    val_dataloader = DataLoader(dataset=val_data, batch_size=cfg.dataset.batch_size,
                                collate_fn=lambda x: val_data.collate_fn(x))

    model = get_model(cfg.model_type, Gin=train_dataset.Gin, Gout=train_dataset.Gout, cache_dir=cache_dir)
    # init network
    model = model.to(device)

    train(model, train_dataloader, val_dataloader, cfg.dataset)


if __name__ == '__main__':
    main()
