import numpy as np


def run(return_values, feed_dict=None, backwards=True):
    feed_dict = feed_dict or {}
    for operation, value in feed_dict.items():
        operation.run(value)

    if isinstance(return_values, (list, tuple)):
        ret = []
        for operation in return_values:
            ret.append(operation.get_output())
            if backwards:
                operation.backwards()
        return tuple(ret)
    elif isinstance(return_values, dict):
        ret = {}
        for key, operation in return_values.items():
            ret[key] = operation.get_output()
            if backwards:
                operation.backwards()
        return ret

    operation = return_values
    ret = operation.get_output()
    if backwards:
        operation.backwards()
    return ret


class ChainedOperation(object):
    def __init__(self, inputs=None):
        self.input_objects = inputs or []
        self.output_objects = []
        self.inputs_ready = {}
        self.outputs_ready = {}
        self.inputs = []
        self.output = None
        self.grad = 0

        for i in self.input_objects:
            if isinstance(i, ChainedOperation) and not isinstance(i, Variable):
                self.inputs_ready[i] = False
                i.add_output(self)

    def add_output(self, output_object):
        self.output_objects.append(output_object)
        self.outputs_ready[output_object] = False

    def forwards(self, input_object):
        self.inputs_ready[input_object] = True

        if self.all_inputs_ready():
            self.inputs.clear()
            for i in self.input_objects:
                if isinstance(i, ChainedOperation):
                    self.inputs.append(i.get_output())
                else:
                    self.inputs.append(i)
            self.output = self.calc_forwards(self.inputs)

            for o in self.output_objects:
                o.forwards(self)

            # Clear variables
            self.grad = 0
            for key in self.outputs_ready.keys():
                self.outputs_ready[key] = False
            for key in self.inputs_ready.keys():
                self.inputs_ready[key] = False

    def backwards(self, output_object=None):
        if len(self.inputs) != len(self.input_objects):
            raise ValueError('cannot preform backwards pass before full forwards pass')

        if output_object is None:
            self.grad = np.ones_like(self.output)
        else:
            self.outputs_ready[output_object] = True
            self.grad += output_object.get_grad(self)

        if self.all_outputs_ready():
            for input_object in self.input_objects:
                if isinstance(input_object, ChainedOperation):
                    input_object.backwards(self)

    def get_output(self):
        return self.output

    def get_grad(self, input_object=None):
        if input_object is None:
            return self.grad
        return np.multiply(self.grad, self.calc_backwards(input_object))

    def all_inputs_ready(self):
        return self.all_ready(self.inputs_ready)

    def all_outputs_ready(self):
        return self.all_ready(self.outputs_ready)

    @staticmethod
    def all_ready(l):
        for state in l.values():
            if state is False:
                return False
        return True

    def calc_forwards(self, inputs):
        raise NotImplementedError('Abstract class')

    def calc_backwards(self, input_object):
        raise NotImplementedError('Abstract class')


class UnaryChainedOperation(ChainedOperation):
    def __init__(self, x):
        super(UnaryChainedOperation, self).__init__([x])

    def calc_forwards(self, inputs):
        return self.calc_forwards_single(inputs[0])

    def calc_backwards(self, input_object):
        return self.calc_backwards_single(self.inputs[0])

    def calc_forwards_single(self, x):
        raise NotImplementedError('Abstract class')

    def calc_backwards_single(self, x):
        raise NotImplementedError('Abstract class')


class BinaryChainedOperation(ChainedOperation):
    def __init__(self, a, b):
        super(BinaryChainedOperation, self).__init__([a, b])

    def calc_forwards(self, inputs):
        return self.calc_forwards_binary(inputs[0], inputs[1])

    def calc_backwards(self, input_object):
        return self.calc_backwards_binary(input_object, self.inputs[0], self.inputs[1])

    def calc_forwards_binary(self, a, b):
        raise NotImplementedError('Abstract class')

    def calc_backwards_binary(self, input_object, a, b):
        raise NotImplementedError('Abstract class')


class Placeholder(ChainedOperation):

    def __init__(self):
        super(Placeholder, self).__init__()
        self.value = None

    def calc_forwards(self, _):
        return self.value

    def calc_backwards(self, _):
        return 1

    def run(self, value):
        self.value = np.asmatrix(value)
        self.forwards(self)


class Variable(Placeholder):
    def __init__(self, shape, random_func=np.random.rand):
        super(Variable, self).__init__()
        self.shape = shape
        self.random_func = random_func
        self.initialize()

    def get_output(self):
        return self.value

    def update(self, delta, direction=1):
        self.value += direction * delta

    def initialize(self):
        self.value = self.random_func(*self.shape)

    def reset_grad(self):
        self.grad = 0


class Log(UnaryChainedOperation):
    def calc_forwards_single(self, x):
        return np.log(x)

    def calc_backwards_single(self, x):
        return 1 / x


class Sum(UnaryChainedOperation):
    def __init__(self, x, axis=None):
        super(Sum, self).__init__(x)
        self.axis = axis

    def calc_forwards_single(self, x):
        return np.sum(x, self.axis)

    def get_grad(self, input_object):
        if self.axis is not None:
            reps = np.ones_like(np.shape(self.inputs[0]))
            reps[self.axis] = np.shape(self.inputs[0])[self.axis]
            return np.tile(self.grad, reps)
        return super(Sum, self).get_grad(input_object)

    def calc_backwards_single(self, x):
        return np.ones_like(x)


class Reciprocal(UnaryChainedOperation):
    def calc_forwards_single(self, x):
        return 1 / x

    def calc_backwards_single(self, x):
        return - 1 / np.square(x)


class Exp(UnaryChainedOperation):
    def __init__(self, x, axis=None):
        super(Exp, self).__init__(x)
        self.axis = axis

    def calc_forwards_single(self, x):
        if self.axis is not None:
            return np.exp(x - np.max(x, axis=self.axis))

        return np.exp(x)

    def calc_backwards_single(self, x):
        if self.axis is not None:
            return np.exp(x - np.max(x, axis=self.axis))

        return np.exp(x)


class Mul(BinaryChainedOperation):
    def calc_forwards_binary(self, a, b):
        return np.multiply(a, b)

    def calc_backwards_binary(self, input_object, a, b):
        if self.input_objects.index(input_object) == 0:
            return b
        return a


class Dot(BinaryChainedOperation):
    def get_grad(self, input_object):
        if self.input_objects.index(input_object) == 0:
            return np.dot(self.grad, np.transpose(self.inputs[1]))
        return np.dot(np.transpose(self.inputs[0]), self.grad)

    def calc_forwards_binary(self, a, b):
        return np.dot(a, b)

    def calc_backwards_binary(self, input_object, a, b):
        pass
