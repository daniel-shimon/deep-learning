import numpy as np


def run(return_values, feed_dict=None):
    feed_dict = feed_dict or {}
    for operation, value in feed_dict.items():
        operation.run(value)

    if isinstance(return_values, (list, tuple)):
        ret = []
        for operation in return_values:
            ret.append(operation.output)
            operation.backwards()
        return tuple(ret)
    elif isinstance(return_values, dict):
        ret = {}
        for key, operation in return_values.items():
            ret[key] = operation.output
            operation.backwards()
        return ret

    operation = return_values
    ret = operation.get_output()
    operation.backwards()
    return ret


class ChainedOperation(object):
    def __init__(self, inputs=None):
        self.input_objects = inputs or []
        self.output_objects = []
        self.inputs_ready = {}
        self.outputs_ready = {}
        self.inputs = None
        self.output = None
        self.grad = 0

        for i in self.input_objects:
            if isinstance(i, ChainedOperation):
                self.inputs_ready[i] = False
                i.add_output(self)

    def add_output(self, output_object):
        self.output_objects.append(output_object)
        self.outputs_ready[output_object] = False

    def forwards(self, input_object):
        self.inputs_ready[input_object] = True

        if self.all_inputs_ready():
            self.inputs = []
            for i in self.input_objects:
                if isinstance(i, ChainedOperation):
                    self.inputs.append(i.get_output())
                else:
                    self.inputs.append(i)
            self.output = self.calc_forwards(self.inputs)

            for o in self.output_objects:
                o.forwards(self)

            # Clear backwards variables
            self.grad = 0
            for key in self.outputs_ready.keys():
                self.outputs_ready[key] = False

    def backwards(self, output_object=None):
        if output_object is None:
            self.grad = 1
        else:
            self.outputs_ready[output_object] = True
            self.grad += output_object.get_grad(self)

        if self.all_outputs_ready():
            for input_object in self.input_objects:
                if isinstance(input_object, ChainedOperation):
                    input_object.backwards(self)

    def get_output(self):
        return self.output

    def get_grad(self, input_object):
        return self.grad * self.calc_backwards(input_object)

    def all_inputs_ready(self):
        for state in self.inputs_ready.values():
            if state is False:
                return False
        return True

    def all_outputs_ready(self):
        for o in self.outputs_ready.items():
            if o is False:
                return False
        return True

    def calc_forwards(self, inputs):
        raise NotImplementedError('Abstract class')

    def calc_backwards(self, input_object):
        raise NotImplementedError('Abstract class')


class SingleInputChainedOperation(ChainedOperation):
    def __init__(self, x):
        super(SingleInputChainedOperation, self).__init__([x])

    def calc_forwards(self, inputs):
        return self.calc_forwards_single(inputs[0])

    def calc_backwards(self, input_object):
        return self.calc_backwards_single(self.inputs[0])

    def calc_forwards_single(self, x):
        raise NotImplementedError('Abstract class')

    def calc_backwards_single(self, x):
        raise NotImplementedError('Abstract class')


class Placeholder(ChainedOperation):

    def __init__(self):
        super(Placeholder, self).__init__()
        self.value = None

    def calc_forwards(self, _):
        return self.value

    def calc_backwards(self, _):
        return np.ones_like(self.value, dtype=np.float64)

    def run(self, value):
        self.value = np.asarray(value)
        self.forwards(self)


class Gradient(ChainedOperation):
    def __init__(self, operation):
        super(Gradient, self).__init__([])
        operation.input_objects.append(self)

    def calc_forwards(self, inputs):
        return 1

    def calc_backwards(self, _):
        return 1

    def backwards(self, output_object=None):
        self.grad = 0
        super(Gradient, self).backwards(output_object)
        self.output = self.grad


class Log(SingleInputChainedOperation):
    def calc_forwards_single(self, x):
        return np.log(x)

    def calc_backwards_single(self, x):
        return 1 / x


class Sum(SingleInputChainedOperation):
    def calc_forwards_single(self, x):
        return np.sum(x)

    def calc_backwards_single(self, x):
        return 1


class Mul(ChainedOperation):
    def __init__(self, a, b):
        super(Mul, self).__init__([a, b])

    def calc_forwards(self, inputs):
        return np.multiply(inputs[0], inputs[1])

    def calc_backwards(self, input_object):
        for i in self.input_objects:
            if i is not input_object:
                if isinstance(i, ChainedOperation):
                    return i.get_output()

                return i


class Dot(ChainedOperation):
    def __init__(self, a, b):
        super(Dot, self).__init__([a, b])

    def calc_forwards(self, inputs):
        return np.dot(inputs[0], inputs[1])

    def calc_backwards(self, input_object):
        for i in self.input_objects:
            if i is not input_object:
                if isinstance(i, ChainedOperation):
                    return i.get_output()

                return i


class Reciprocal(SingleInputChainedOperation):
    def calc_forwards_single(self, x):
        return 1 / x

    def calc_backwards_single(self, x):
        return - 1 / np.square(x)


class Exp(SingleInputChainedOperation):
    def calc_forwards_single(self, x):
        if hasattr(x, 'max'):
            return np.exp(x - x.max())

        return np.exp(x)

    def calc_backwards_single(self, x):
        if hasattr(x, 'max'):
            return np.exp(x - x.max())

        return np.exp(x)
