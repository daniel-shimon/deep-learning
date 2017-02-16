"""
Naive optimizers with numeric gradient
"""


import math
import numpy as np


def numeric_gradient(x, f):
    x = np.asmatrix(x, np.float64)
    d_x = np.zeros_like(x)
    fx = f(x)
    r = [None]
    if np.shape(fx) != () and np.shape(fx) != np.shape(d_x):
        for rank in range(len(np.shape(fx))):
            if np.shape(fx)[rank] != np.shape(d_x)[rank]:
                r[0] = rank
                break

    it = np.nditer(d_x, flags=['multi_index'])
    while not it.finished:
        like_x = np.copy(x)
        dx = like_x[it.multi_index] * 1e-8
        like_x[it.multi_index] += dx

        derivative = (f(like_x) - fx) / dx
        if np.shape(derivative) == ():
            d_x[it.multi_index] = derivative
        elif np.shape(derivative) == np.shape(d_x):
            d_x[it.multi_index] = derivative[it.multi_index]
        elif r[0] is not None:
            for i in range(np.shape(derivative)[r[0]]):
                location = list(it.multi_index)
                location[r[0]] = i
                d_x[it.multi_index] += derivative[tuple(location)]

        it.iternext()
    return d_x


class BaseOptimizer(object):
    # noinspection PyMethodMayBeStatic
    def optimize(self, params):
        yield NotImplementedError()


class SGD(BaseOptimizer):
    def __init__(self, function, learning_rate=0.01, gradient_func=numeric_gradient):
        self.function = function
        self.learning_rate = learning_rate
        self.gradient_func = gradient_func

    def optimize(self, params):
        d_params = self.gradient_func(params, self.function)
        for i in range(len(params)):
            params[i] -= self.learning_rate * d_params[i]


class Momentum(BaseOptimizer):
    def __init__(self, function, learning_rate=0.01, gradient_func=numeric_gradient, momentum_strength=0.9):
        self.function = function
        self.learning_rate = learning_rate
        self.gradient_func = gradient_func
        self.momentum_strength = momentum_strength
        self.last_v = 0

    def optimize(self, params):
        d_params = self.gradient_func(params, self.function)
        for i in range(len(params)):
            v = self.momentum_strength*self.last_v + self.learning_rate*d_params[i]
            params[i] -= v
            self.last_v = v


class NEG(BaseOptimizer):
    """
        Nestrov Accelerated Gradient
    """
    def __init__(self, function, learning_rate=0.01, gradient_func=numeric_gradient, momentum_strength=0.9):
        self.function = function
        self.learning_rate = learning_rate
        self.gradient_func = gradient_func
        self.momentum_strength = momentum_strength
        self.last_v = 0

    def optimize(self, params):
        d_params = self.gradient_func([param - self.momentum_strength*self.last_v for param in params],
                                      self.function)
        for i in range(len(params)):
            v = self.momentum_strength*self.last_v + self.learning_rate*d_params[i]
            params[i] -= v
            self.last_v = v


class Adagrad(BaseOptimizer):
    def __init__(self, function, learning_rate=0.01, gradient_func=numeric_gradient):
        self.function = function
        self.learning_rate = learning_rate
        self.gradient_func = gradient_func
        self.G = []

    def optimize(self, params):
        if not len(self.G) == len(params):
            self.G = [0 for _ in range(len(params))]
        d_params = self.gradient_func(params,
                                      self.function)
        for i in range(len(params)):
            g = d_params[i]
            self.G[i] += g ** 2
            params[i] -= (self.learning_rate / math.sqrt(self.G[i] + 1e-8)) * g


class Adadelta(BaseOptimizer):
    def __init__(self, function, decay=0.9, gradient_func=numeric_gradient):
        self.function = function
        self.decay = decay
        self.gradient_func = gradient_func
        self.decaying_g = []
        self.decaying_d = []

    def optimize(self, params):
        if not len(self.decaying_g) == len(params):
            self.decaying_g = [0 for _ in range(len(params))]
        if not len(self.decaying_d) == len(params):
            self.decaying_d = [0 for _ in range(len(params))]

        d_params = self.gradient_func(params,
                                      self.function)
        for i in range(len(params)):
            g = d_params[i]
            self.decay_update(self.decaying_g, i, g)
            d = -(math.sqrt(self.decaying_d[i] + 1e-8) / math.sqrt(self.decaying_g[i] + 1e-8)) * g
            params[i] += d
            self.decay_update(self.decaying_d, i, d)

    def decay_update(self, l, i, value):
        l[i] = self.decay*l[i] + (1 - self.decay) * (value ** 2)


class RMSProp(BaseOptimizer):
    def __init__(self, function, learning_rate=0.001, decay=0.9, gradient_func=numeric_gradient):
        self.function = function
        self.learning_rate = learning_rate
        self.decay = decay
        self.gradient_func = gradient_func
        self.decaying_g = []

    def optimize(self, params):
        if not len(self.decaying_g) == len(params):
            self.decaying_g = [0 for _ in range(len(params))]

        d_params = self.gradient_func(params,
                                      self.function)
        for i in range(len(params)):
            g = d_params[i]
            self.decay_update(self.decaying_g, i, g)
            d = -(self.learning_rate / math.sqrt(self.decaying_g[i] + 1e-8)) * g
            params[i] += d

    def decay_update(self, l, i, value):
        l[i] = self.decay*l[i] + (1 - self.decay) * (value ** 2)


class Adam(BaseOptimizer):
    def __init__(self, function, learning_rate=0.001, decay1=0.9, decay2=0.999, gradient_func=numeric_gradient):
        self.function = function
        self.learning_rate = learning_rate
        self.decay1 = decay1
        self.decay2 = decay2
        self.gradient_func = gradient_func
        self.decaying_v = []
        self.decaying_m = []

    def optimize(self, params):
        if not len(self.decaying_v) == len(params):
            self.decaying_v = [0 for _ in range(len(params))]
        if not len(self.decaying_m) == len(params):
            self.decaying_m = [0 for _ in range(len(params))]

        d_params = self.gradient_func(params,
                                      self.function)
        for i in range(len(params)):
            self.decay_updates(i, d_params[i])
            m_hat = self.decaying_m[i] / (1 - self.decay1)
            v_hat = self.decaying_v[i] / (1 - self.decay2)
            d = -(self.learning_rate / (math.sqrt(v_hat) + 1e-8)) * m_hat
            params[i] += d

    def decay_updates(self, i, value):
        self.decaying_m[i] = self.decay1 * self.decaying_m[i] + (1 - self.decay1) * value
        self.decaying_v[i] = self.decay2 * self.decaying_v[i] + (1 - self.decay2) * (value ** 2)
