import pathlib
from collections.abc import Callable
from os import PathLike

import numpy as np
from omegaconf import OmegaConf


def load_config_hierarchy(cfg_path: PathLike):
    p = pathlib.Path(cfg_path)
    assert p.exists(), f"Cfg path {p.absolute()} does not exist"
    cfg = OmegaConf.load(p)
    if "defaults" in cfg:
        parent_cfg_names = list(cfg.defaults)
        if "_self_" in parent_cfg_names:
            parent_cfg_names.remove("_self_")
        for parent_cfg_name in parent_cfg_names:
            parent_path = p.parent / f"{parent_cfg_name}.yaml"
            parent_cfg = load_config_hierarchy(parent_path)
            cfg = OmegaConf.merge(parent_cfg, cfg)
    return cfg


class CallableDict(dict, Callable):
    def __call__(self, key):
        return self[key]


def flatten_dict(d: dict, prefix=""):
    a = {}
    for k, v in d.items():
        if isinstance(v, dict):
            a.update(flatten_dict(v, prefix=f"{k}/"))
        else:
            a[f"{prefix}{k}"] = v
    return a


class TemporaryNumpySeed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, *args):
        np.random.set_state(self.state)


class ConfigException(Exception):
    """Exception raised for errors in the configuration."""

    def __init__(self, message):
        super().__init__(message)
