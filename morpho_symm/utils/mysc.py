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


def compare_dictionaries(dict1: dict, dict2: dict):
    """Recursively compare dictionaries which entries might be other dictionaries.
    Any entry in one dictionary that is not present in the other dictionary needs to be reported.

    Args:
        dict1: A tree of dictionaries.
        dict2: A tree of dictionaries.

    Returns: The dictionary containing only the keys that are different between the two dictionaries. In all levels of
    the tree. Each entry of this difference dictionary is a tuple (val1, val2) when val1 != val2.

    """
    diff = {}
    for key in dict1.keys():
        if key not in dict2.keys():
            diff[key] = (dict1[key], None)
        else:
            if isinstance(dict1[key], dict):
                inner_diff = compare_dictionaries(dict1[key], dict2[key])
                if len(inner_diff) > 0:
                    diff[key] = inner_diff
            elif isinstance(dict1[key], np.ndarray):
                if not np.allclose(dict1[key] - dict2[key], 0, rtol=1e-6, atol=1e-6):
                    diff[key] = (dict1[key], dict2[key])
            else:
                if dict1[key] != dict2[key]:
                    diff[key] = (dict1[key], dict2[key])

    for key in dict2.keys():
        if key not in dict1.keys():
            diff[key] = (None, dict2[key])

    return diff


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
