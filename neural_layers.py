import numpy as np

import chained_operations as op


class Layer(op.ChainedOperation):
    def __init__(self, inputs=None):
        inputs = inputs or []
        super(Layer, self).__init__(inputs)
        self.is_locked = False

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
            if self.last != self:
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

    def calc_backwards(self, input_object):
        pass

    def get_variables(self):
        if self.is_locked:
            return []
        return self.variables

    def lock(self):
        self.is_locked = True

    def unlock(self):
        self.is_locked = False

    def set_inputs(self, new_inputs=None):
        new_inputs = new_inputs or []
        if not len(new_inputs) == len(self.input_objects):
            raise ValueError('cannot change inputs number')

        self.input_objects = new_inputs or []
        self.inputs_ready = {}
        for i in self.input_objects:
            if isinstance(i, op.ChainedOperation):
                if not isinstance(i, op.Variable):
                    self.inputs_ready[i] = False
                i.add_output(self)

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


class VariableLayer(op.Variable, Layer):
    def __init__(self, depth=0, size=0):
        self.size = size
        super(VariableLayer, self).__init__((depth, size))

    def get_output_len(self):
        return self.size

    def build_layer(self, inputs):
        return self, self


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

    def set_inputs(self, new_input=None):
        super(UnaryLayer, self).set_inputs([new_input])


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
        b = op.Variable((1, self.neurons))
        y = op.Add(op.Dot(x, w), b, axis=0)
        return y, [w, b]

    def get_unary_len(self, shape):
        return self.neurons
