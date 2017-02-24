import numpy as np
import itertools as it

import chained_operations as op
import neural_layers as nn
import data_utils
import plotting
import chained_optimizers as optimizers

print('get data')

label_names = data_utils.Cifar10.load_labels()

data, labels = data_utils.Cifar10.get_batch(1, raw=False)

print('build model')

x = nn.InputLayer(1024 * 3)

y = nn.Dense(x, 10)

y_ = op.Placeholder()
loss = nn.Softmax(y, y_)

print('train model')

batch_size = 100

graph = plotting.Graph('loss')
optimizer = optimizers.Adagrad(loss)

for count, mini_batch_x, mini_batch_y_ in zip(
        it.count(),
        data_utils.batches(data, batch_size),
        data_utils.batches(labels, batch_size)):

    batch_loss = optimizer.step({x: mini_batch_x, y_: mini_batch_y_})
    batch_loss = np.mean(batch_loss)

    print('%d loss - %f' % (count, batch_loss))
    graph.maybe_add(count, count, batch_loss, True)

    if batch_loss < 4:
        break


print('deep dream')

images = nn.VariableLayer(1, 1024 * 3)
y.set_inputs(images)
y.lock()

y.set_inputs(x)

truths = np.zeros((1, 10))
truths[0][1] = 1

graph.clear()
optimizer = optimizers.Adagrad(loss)

for i in range(15):
    loss = np.mean(optimizer.step({y_: truths}))
    print('%d loss - %f' % (i, loss))
    graph.maybe_add(i, i, loss, plot=True)

print('done')
