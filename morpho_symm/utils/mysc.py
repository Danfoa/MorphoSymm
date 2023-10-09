from collections.abc import Callable


class CallableDict(dict, Callable):

    def __call__(self, key):
        return self[key]
