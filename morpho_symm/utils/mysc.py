from collections.abc import Callable


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