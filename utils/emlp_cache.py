
from emlp.reps.representation import Base
import itertools


class EMLPCache(dict):

    def __init__(self, cache=None, lazy_cache=None):
        super().__init__()
        self.cache = cache if cache else {}
        self.lazy_cache = lazy_cache if lazy_cache else {}

    def items(self): # real signature unknown; restored from __doc__
        return itertools.chain(self.cache.items(), self.lazy_cache.items())

    def keys(self): # real signature unknown; restored from __doc__
        return itertools.chain(self.cache.keys(), self.lazy_cache.keys())

    def values(self): # real signature unknown; restored from __doc__
        return itertools.chain(self.cache.values(), self.lazy_cache.values())

    def __contains__(self, item):
        return str(item) in self.lazy_cache or item in self.cache

    def __getitem__(self, y):
        # Search first in lazy cache then in running cache
        if str(y) in self.lazy_cache:
            return self.lazy_cache[str(y)]
        elif isinstance(y, Base):
            return self.cache[y]
        else:
            raise KeyError(y)

    def __setitem__(self, k, v):
        """ Set self[key] to value. """
        if isinstance(k, Base):
            self.cache[k] = v
        elif isinstance(k, str):
            self.lazy_cache[k] = v

    def __len__(self):
        return len(self.cache) + len(self.lazy_cache)