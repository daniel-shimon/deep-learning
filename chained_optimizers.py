import chained_operations as op
import neural_layers as nn


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
            self.update(variable, variable.get_grad())
        return loss

    def reset_grads(self):
        for variable in self.variables:
            variable.reset_grad()

    def update(self, variables, gradients):
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
    def __init__(self, loss, learning_rate=0.01, minimize=True):
        super(SGD, self).__init__(loss, minimize)
        self.learning_rate = learning_rate

    def update(self, variable, gradient):
        variable.update(self.learning_rate * gradient, direction=self.direction)
