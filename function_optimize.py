import math
import random
import itertools as it
import plotting
import optimizers


def func(params):
    x = params[0]

    return abs(6 - abs(math.sin(x*0.5)*math.log1p(x**2)))

optimizer_list = [optimizers.Adadelta(func), optimizers.Adagrad(func), optimizers.Adam(func),
                  optimizers.Momentum(func), optimizers.NEG(func), optimizers.RMSProp(func),
                  optimizers.SGD(func)]

params = []
graphs = []
p = random.randint(-200, 200)
for optimizer in optimizer_list:
    graphs.append(plotting.Graph(optimizer.__class__.__name__))
    params.append([p])

for i in it.count():
    for param, graph, optimizer in zip(params, graphs, optimizer_list):
        f = func(param)
        graph.maybe_add(i, i, f)
        optimizer.optimize(param)
    plotting.replot()
