from collections.abc import Callable

import numpy as np


class CallableDict(dict, Callable):

    def __call__(self, key):
        return self[key]

def flatten_dict(d: dict, prefix=''):
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