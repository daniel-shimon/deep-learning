import chained_operations as op
import neural_layers as nn
import numpy as np


class ChainedOptimizer(object):
    def __init__(self, loss, minimize=True):
        self.loss = loss
        self.direction = -1 if minimize else 1

        self.variables = self.find_variables(loss)
        if not self.variables:
            raise ValueError('no variables found to optimize')

    def step(self, feed_dict):
        self.reset_grads()
        loss = op.run(self.loss, feed_dict)

        for variable in self.variables:
            variable.update(self.update(variable, variable.get_grad()), self.direction)
        return loss

    def reset_grads(self):
        for variable in self.variables:
            variable.reset_grad()

    def update(self, variable, gradient):
        raise NotImplementedError()

    @staticmethod
    def find_variables(chained_operation):
        if isinstance(chained_operation, op.ChainedOperation):
            if isinstance(chained_operation, op.Variable):
                return [chained_operation]

            variables = []
            if isinstance(chained_operation, nn.Layer):
                variables.extend(chained_operation.get_variables())
            for input_object in chained_operation.input_objects:
                variables.extend(ChainedOptimizer.find_variables(input_object))
            return variables
        return []


class SGD(ChainedOptimizer):
    def __init__(self, loss, learning_rate=0.01):
        super(SGD, self).__init__(loss)
        self.learning_rate = learning_rate

    def update(self, _, gradient):
        return self.learning_rate * gradient


class Momentum(SGD):
    def __init__(self, loss, learning_rate=0.01, momentum=0.9):
        super(Momentum, self).__init__(loss, learning_rate)
        self.momentum = momentum
        self.last_change = {variable: 0 for variable in self.variables}

    def update(self, variable, gradient):
        change = self.momentum * self.last_change[variable] + self.learning_rate * gradient
        self.last_change[variable] = change
        return self.last_change[variable]


class Adagrad(SGD):
    def __init__(self, loss, learning_rate=0.01):
        super(Adagrad, self).__init__(loss, learning_rate)
        self.squares_sum = {variable: 0 for variable in self.variables}

    def update(self, variable, gradient):
        self.squares_sum[variable] += np.square(gradient)
        return np.multiply(self.learning_rate / np.sqrt(self.squares_sum[variable] + 1e-8), gradient)
