"""
Naive operations with forwards and backwards numpy functions
"""


import numpy as np


class Operation(object):
    def forwards(self, *args):
        yield NotImplementedError('Abstract class')

    def backwards(self, *args):
        yield NotImplementedError('Abstract class')


class Sum(Operation):
    # noinspection PyAttributeOutsideInit
    def forwards(self, x):
        self.x = x
        return np.sum(x)

    def backwards(self):
        return np.ones_like(self.x)


class Dot(Operation):
    # noinspection PyAttributeOutsideInit
    def forwards(self, x, a):
        self.a = a
        return np.dot(a, x)

    def backwards(self):
        return self.a


class Reciprocal(Operation):
    # noinspection PyAttributeOutsideInit
    def forwards(self, x):
        self.x = x
        return 1 / x

    def backwards(self):
        return - 1 / np.square(self.x)


class Exp(Operation):
    # noinspection PyAttributeOutsideInit
    def forwards(self, x):
        self.x = x
        return np.exp(x)

    def backwards(self):
        return np.exp(self.x)
