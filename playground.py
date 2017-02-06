import chained_operations as op
import optimizers
import numpy as np


def check_operation(forwards, backwards):
    x = np.array([0.1, 0.3])
    forwards(x)
    grad = backwards()
    n_grad = optimizers.numeric_gradient(x, forwards)
    print (grad, n_grad)

y_ = np.array([1, 0])

x = op.Variable()

y = op.Sum(op.Exp(x))

g = op.Gradient(x)

print op.run([g, y], {x: np.array([1, 2, 3])})

print optimizers.numeric_gradient(np.array([1,2,3]), lambda inp: op.run(y, {x: inp}))