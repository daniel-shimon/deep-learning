import copy
import math
import numpy as np


def numeric_gradient(params, f):
    params = np.asarray(params)
    d_params = np.zeros_like(params)
    fx = f(params)
    for i in xrange(params.shape[0]):
        like_params = np.copy(params)
        dx = 1e-9
        like_params[i, ] += dx

        derivative = (f(like_params) - fx) / dx
        if np.shape(derivative) == ():
            d_params[i, ] = derivative
        else:
            d_params[i, ] = derivative[i, ]
    return d_params


class BaseOptimizer(object):
    # noinspection PyMethodMayBeStatic
    def optimize(self, params):
        yield NotImplementedError('Abstract class')


class SGD(BaseOptimizer):
    def __init__(self, function, learning_rate=0.01, gradient_func=numeric_gradient):
        self.function = function
        self.learning_rate = learning_rate
        self.gradient_func = gradient_func

    def optimize(self, params):
        d_params = self.gradient_func(params, self.function)
        for i in xrange(len(params)):
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
        for i in xrange(len(params)):
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
        for i in xrange(len(params)):
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
            self.G = [0 for _ in xrange(len(params))]
        d_params = self.gradient_func(params,
                                      self.function)
        for i in xrange(len(params)):
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
            self.decaying_g = [0 for _ in xrange(len(params))]
        if not len(self.decaying_d) == len(params):
            self.decaying_d = [0 for _ in xrange(len(params))]

        d_params = self.gradient_func(params,
                                      self.function)
        for i in xrange(len(params)):
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
            self.decaying_g = [0 for _ in xrange(len(params))]

        d_params = self.gradient_func(params,
                                      self.function)
        for i in xrange(len(params)):
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
            self.decaying_v = [0 for _ in xrange(len(params))]
        if not len(self.decaying_m) == len(params):
            self.decaying_m = [0 for _ in xrange(len(params))]

        d_params = self.gradient_func(params,
                                      self.function)
        for i in xrange(len(params)):
            self.decay_updates(i, d_params[i])
            m_hat = self.decaying_m[i] / (1 - self.decay1)
            v_hat = self.decaying_v[i] / (1 - self.decay2)
            d = -(self.learning_rate / (math.sqrt(v_hat) + 1e-8)) * m_hat
            params[i] += d

    def decay_updates(self, i, value):
        self.decaying_m[i] = self.decay1 * self.decaying_m[i] + (1 - self.decay1) * value
        self.decaying_v[i] = self.decay2 * self.decaying_v[i] + (1 - self.decay2) * (value ** 2)
