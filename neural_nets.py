import chained_operations as op


class Layer(op.ChainedOperation):
    def __init__(self, inputs=None):
        super(Layer, self).__init__(inputs)
        self.firsts = [op.Placeholder() for _ in inputs]
        self.last = self.build_layer(self.firsts)

    def calc_forwards(self, inputs):
        for i, input_value in enumerate(inputs):
            self.firsts[i].run(input_value)
        return self.last.output

    def calc_backwards(self, input_object):
        self.last.backwards()
        index = self.input_objects.index(input_object)
        return self.firsts[index].grad

    def build_layer(self, inputs):
        raise NotImplementedError()


class UnaryLayer(Layer):
    def __init__(self, x):
        super(UnaryLayer, self).__init__([x])

    def build_layer(self, inputs):
        return self.build_binary_layer(inputs[0])

    def build_binary_layer(self, x):
        raise NotImplementedError()


class BinaryLayer(Layer):
    def __init__(self, a, b):
        super(BinaryLayer, self).__init__([a, b])

    def build_layer(self, inputs):
        return self.build_binary_layer(inputs[0], inputs[1])

    def build_binary_layer(self, a, b):
        raise NotImplementedError()


class Softmax(BinaryLayer):
    def __init__(self, y, y_):
        super(Softmax, self).__init__(y, y_)

    def build_binary_layer(self, y, y_):
        exp = op.Exp(y, axis=1)
        top = op.Sum(op.Mul(y_, exp), axis=1)
        reciprocal = op.Reciprocal(op.Sum(exp, axis=1))
        naive = op.Mul(top, reciprocal)
        return op.Mul(-1, op.Log(naive))
