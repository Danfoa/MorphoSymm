#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/2/22
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com

from emlp.groups import Group, Trivial
import jax.numpy as jnp
import numpy as np
from emlp.reps.linear_operators import lazify, LazyKron, densify
from emlp.reps.representation import Vector


# class autoVector(Vector):
#
#     def __init__(self, G=None):
#         super(autoVector, self).__init__(G)
#
#         if G is not None:
#             # Try loading solcache
#             pass
#
#     def save(self, ):
#         pass


class SemiDirectProduct(Group):
    """SemiDirectProduct"""

    def __init__(self, Gin: Group, Gout: Group):

        self.G1 = Gin
        self.G2 = Gout
        assert len(Gin.discrete_generators) == len(Gout.discrete_generators)

        self.discrete_generators = []
        for h_in, h_out in zip(Gin.discrete_generators, Gout.discrete_generators):
            a = LazyKron([h_out, h_in])
            self.discrete_generators.append(densify(a))
        self.discrete_generators = np.array(self.discrete_generators)
        super().__init__()

    def get_inout_generators(self):
        return np.array(self.G1.discrete_generators, dtype=np.float32), \
               np.array(self.G2.discrete_generators, dtype=np.float32)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        outstr = f'({repr(self.G1)})⋊({self.G2})'
        if self.args:
            outstr += '(' + ''.join(repr(arg) for arg in self.args) + ')'
        return outstr