import cPickle
import numpy
import os
import itertools
from PIL import Image


with open(os.path.join('cifar-10-batches-py', 'batches.meta'), 'rb') as f:
    label_names = cPickle.load(f)['label_names']


def get_batch(n):
    path = os.path.join('cifar-10-batches-py', 'data_batch_%d' % n)
    with open(path, 'rb') as f:
        dictionary = cPickle.load(f)
    return dictionary


def batches(data, batch_size):
    for i in xrange(0, len(data), batch_size):
        yield data[i:i + batch_size]


def show_img_array(flat):
    img_array = numpy.zeros((32, 32, 3))
    for channel in range(3):
        for row in range(32):
            for collum in range(32):
                img_array[row, collum, channel] = flat[channel * 1024 + row * 32 + collum]
    with Image.fromarray(img_array.astype(numpy.uint8)) as img:
        img.show()


def forward_propagation(inputs, weights):
    return numpy.matmul(inputs, weights)


def svm(outputs, truth_index):
    margins = numpy.maximum(0, outputs - outputs[truth_index] + 1)
    margins[truth_index] = 0
    return numpy.sum(margins)


def softmax(outputs, truth_index):
    n_outputs = outputs - numpy.max(outputs)
    return -numpy.log(numpy.exp(n_outputs[truth_index]) / numpy.sum(numpy.exp(n_outputs)))


def regularize(w):
    return numpy.sum(w ** 2)


def loss(x, y_, w):
    l = []
    for inputs, truth in zip(x, y_):
        l.append(softmax(forward_propagation(inputs, w), truth) + 1e-4 * regularize(w))
    return numpy.mean(l)


def numerical_gradient(x, y_, w, f):
    d_weigths = numpy.zeros_like(w)
    for i in xrange(w.shape[0]):
        for j in xrange(w.shape[1]):
            test_weights = numpy.array(w)
            dx = 1e-8
            test_weights[i][j] += dx
            d_weigths[i][j] = (loss(x, y_, test_weights) - f) / dx
    return d_weigths


def train_batch(x, y_, w, learning_rate):
    eval = loss(x, y_, w)
    d = numerical_gradient(x, y_, w, eval)
    w -= d * learning_rate

w = numpy.random.randn(1024 * 3, 10) * 1e-4

batch = get_batch(1)
data, labels = batch['data'], batch['labels']

learning_rate = 0.001
batch_size = 2

counter = 0
for mini_batch_x, mini_batch_y_ in itertools.izip(
        batches(data, batch_size),
        batches(labels, batch_size)):
    print '%d: loss = %f' % (counter, loss(mini_batch_x, mini_batch_y_, w))
    train_batch(mini_batch_x, mini_batch_y_, w, learning_rate)
    counter += 1
