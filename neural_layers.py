import numpy as np

import chained_operations as op


class Layer(op.ChainedOperation):
    def __init__(self, inputs=None):
        inputs = inputs or []
        super(Layer, self).__init__(inputs)

        self.firsts = [op.Placeholder() for _ in inputs]
        self.input_shapes = []
        for i in inputs:
            if isinstance(i, Layer):
                self.input_shapes.append(i.get_output_len())
            else:
                self.input_shapes.append(None)

        self.last, self.variables = self.build_layer(self.firsts)
        self.variables = self.variables or []
        if not isinstance(self.variables, (list, tuple)):
            self.variables = [self.variables]

    def backwards(self, output_object=None):
        if self.last is not None:
            if output_object is not None:
                self.last.grad = output_object.get_grad(self)
            self.last.backwards()
        super(Layer, self).backwards(output_object)

    def calc_forwards(self, inputs):
        for i, input_value in enumerate(inputs):
            self.firsts[i].run(input_value)
        if self.last is not None:
            return self.last.output

    def get_grad(self, input_object=None):
        if input_object is None:
            return self.grad
        index = self.input_objects.index(input_object)
        return self.firsts[index].grad

    def get_variables(self):
        return self.variables

    def get_output_len(self):
        raise NotImplementedError()

    def build_layer(self, inputs):
        raise NotImplementedError()


class InputLayer(op.Placeholder, Layer):
    def __init__(self, size=0, axis=1):
        self.size = size
        self.axis = axis
        super(InputLayer, self).__init__()

    def build_layer(self, inputs):
        return None, None

    def get_output_len(self):
        return self.size

    def run(self, value):
        shape = np.shape(value)
        if len(shape) < self.axis + 1 or shape[self.axis] != self.size:
            raise ValueError('value shape not matching size %d in axis %d' % (self.size, self.axis))
        super(InputLayer, self).run(value)


class UnaryLayer(Layer):
    def __init__(self, x):
        super(UnaryLayer, self).__init__([x])

    def build_layer(self, inputs):
        return self.build_unary_layer(inputs[0])

    def build_unary_layer(self, x):
        raise NotImplementedError()

    def get_output_len(self):
        return self.get_unary_len(self.input_shapes[0])

    def get_unary_len(self, shape):
        raise NotImplementedError()


class BinaryLayer(Layer):
    def __init__(self, a, b):
        super(BinaryLayer, self).__init__([a, b])

    def build_layer(self, inputs):
        return self.build_binary_layer(inputs[0], inputs[1])

    def build_binary_layer(self, a, b):
        raise NotImplementedError()

    def get_output_len(self):
        return self.get_binary_len(self.input_shapes[0], self.input_shapes[1])

    def get_binary_len(self, shape_a, shape_b):
        raise NotImplementedError()


class Softmax(BinaryLayer):
    def __init__(self, y, y_):
        super(Softmax, self).__init__(y, y_)

    def build_binary_layer(self, y, y_):
        exp = op.Exp(y, axis=1)
        top = op.Sum(op.Mul(y_, exp), axis=1)
        reciprocal = op.Reciprocal(op.Sum(exp, axis=1))
        naive = op.Mul(top, reciprocal)
        return op.Mul(-1, op.Log(naive)), None

    def get_binary_len(self, shape_a, shape_b):
        return 1


class Dense(UnaryLayer):
    def __init__(self, x=None, neurons=1):
        self.neurons = neurons
        super(Dense, self).__init__(x)

    def build_unary_layer(self, x):
        if self.input_shapes[0] is None:
            raise ValueError('input has no defined shape')
        w = op.Variable((self.input_shapes[0], self.neurons))
        return op.Dot(x, w), w

    def get_unary_len(self, shape):
        return self.neurons
