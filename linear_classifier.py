import numpy as np
import itertools as it

import chained_operations as op
import data_utils
import plotting
from optimizers import numeric_gradient

label_names = data_utils.Cifar10.load_labels()

data, labels = data_utils.Cifar10.get_batch(1, raw=False)

x = op.Placeholder()

w = op.Placeholder()
g = op.Gradient(w)

y = op.Dot(x, w)

y_ = op.Placeholder()
loss = op.Mul(-1, op.Log(op.Mul(op.Sum(op.Mul(y_, op.Exp(y, axis=1)), axis=1), op.Reciprocal(op.Sum(op.Exp(y, axis=1), axis=1)))))

batch_size = 10
w_values = np.random.rand(1024 * 3, 10)
graph = plotting.Graph('loss')

for count, mini_batch_x, mini_batch_y_ in zip(
        it.count(),
        data_utils.batches(data, batch_size),
        data_utils.batches(labels, batch_size)):

    batch_loss = np.mean(op.run(loss, {x: mini_batch_x, w: w_values, y_: mini_batch_y_}))
    batch_grad = op.run(g)

    print('%d loss - %f' % (count, batch_loss))
    graph.maybe_add(count, count, batch_loss, True)
    w_values -= batch_grad * 0.01
