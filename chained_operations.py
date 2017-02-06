import numpy as np


def run(return_values, feed_dict):
    for operation, value in feed_dict.items():
        operation.run(value)

    if isinstance(return_values, (list, tuple)):
        ret = []
        for operation in return_values:
            ret.append(operation.output)
        return tuple(ret)
    elif isinstance(return_values, dict):
        ret = {}
        for key, operation in return_values.items():
            ret[key] = operation.output
        return ret

    operation = return_values
    return operation.output


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
            self.inputs_ready[i] = False
            i.add_output(self)

    def add_output(self, output_object):
        self.output_objects.append(output_object)
        self.outputs_ready[output_object] = False

    def forwards(self, input_object):
        self.inputs_ready[input_object] = True

        if self.all_inputs_ready():
            self.inputs = [i.output for i in self.input_objects]
            self.output = self.calc_forwards(self.inputs)

            for o in self.output_objects:
                o.forwards(self)

    def backwards(self, output_object=None):
        if output_object is None:
            self.grad = 1
        else:
            self.outputs_ready[output_object] = True
            self.grad += output_object.grad

        if self.all_outputs_ready():
            self.grad *= self.calc_backwards()

            for input_object in self.input_objects:
                input_object.backwards(self)

    def all_inputs_ready(self):
        for state in self.inputs_ready.items():
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

    def calc_backwards(self):
        raise NotImplementedError('Abstract class')


class Variable(ChainedOperation):

    def __init__(self):
        super(Variable, self).__init__()
        self.value = None

    def calc_forwards(self, _):
        return self.value

    def calc_backwards(self):
        return 1

    def run(self, value):
        self.value = value
        self.forwards(self)


class Gradient(ChainedOperation):
    def __init__(self, operation):
        super(Gradient, self).__init__([operation])

    def calc_forwards(self, inputs):
        return self.input_objects[0].grad

    def calc_backwards(self):
        return 1


class Sum(ChainedOperation):
    def __init__(self, x):
        super(Sum, self).__init__([x])

    def calc_forwards(self, inputs):
        return np.sum(inputs[0])

    def calc_backwards(self):
        return 1


class Exp(ChainedOperation):
    def __init__(self, x):
        super(Exp, self).__init__([x])

    def calc_forwards(self, inputs):
        return np.exp(inputs[0])

    def calc_backwards(self):
        return np.exp(self.inputs[0])
