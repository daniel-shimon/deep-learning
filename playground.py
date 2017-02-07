import chained_operations as op
import optimizers
import plotting
import numpy as np
import itertools as it

y_ = np.array([1, 0])

x = op.Placeholder()
g = op.Gradient(x)

y = op.Mul(-1, op.Log(op.Mul(op.Sum(op.Mul(y_, op.Exp(x))), op.Reciprocal(op.Sum(op.Exp(x))))))
#y = op.Sum(op.Mul(-3, x))


def f(inp):
    return op.run(y, {x: inp})


def grad(inp, _):
    op.run(y, {x: inp})
    return op.run(g)

data1 = np.random.rand(*y_.shape)
data2 = np.copy(data1)
data2[1] += 1

graph1 = plotting.Graph('numeric')
graph2 = plotting.Graph('backprop')

opt1 = optimizers.NEG(f)
opt2 = optimizers.NEG(f, gradient_func=grad)

for i in it.count():
    evl = f(data1)
    graph1.maybe_add(i, i, evl)
    opt1.optimize(data1)

    evl = f(data2)
    graph2.maybe_add(i, i, evl)
    opt2.optimize(data2)

    plotting.replot()
