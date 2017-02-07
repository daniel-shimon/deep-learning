import numpy as np

import chained_operations as op
import data_utils

label_names = data_utils.Cifar10.load_labels()

batch = data_utils.Cifar10.get_batch(1)
data, labels = batch[b'data'], batch[b'labels']

x = op.Placeholder()

w = op.Placeholder()
g = op.Gradient(w)

y = op.Dot(x, w)

y_ = op.Placeholder()
loss = op.Mul(-1, op.Log(op.Mul(op.Dot(y_, op.Exp(y)), op.Reciprocal(op.Sum(op.Exp(y))))))

batch_size = 10
w_values = np.random.randn(1024 * 3, 10) * 1e-4

counter = 0
for mini_batch_x, mini_batch_y_ in zip(
        data_utils.batches(data, batch_size),
        data_utils.batches(labels, batch_size)):
    batch_loss = 0
    batch_grad = 0
    for i in range(batch_size):
        batch_loss += op.run(loss, feed_dict={x: mini_batch_x[i],
                                              y_: mini_batch_y_[i],
                                              w: w_values})
        batch_grad += op.run(g)

    batch_loss /= batch_size
    batch_grad /= batch_size

    print('%d loss - %f' % (counter, batch_loss))
    w_values -= batch_grad * 0.01
    counter += 1
