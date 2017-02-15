import sys
import os
import numpy as np
from PIL import Image

if sys.version_info[0] == 2:
    # noinspection PyUnresolvedReferences,PyPep8Naming
    import cPickle as c_pickle
else:
    import _pickle as c_pickle


def batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


class Cifar10(object):
    @staticmethod
    def load_labels():
        with open(os.path.join('cifar-10-batches-py', 'batches.meta'), 'rb') as f:
            label_names = c_pickle.load(f)['label_names']
        return label_names

    @staticmethod
    def get_batch(n, raw=False):
        path = os.path.join('cifar-10-batches-py', 'data_batch_%d' % n)
        with open(path, 'rb') as f:
            dictionary = c_pickle.load(f, encoding='bytes')
        if not raw:
            dictionary[b'data'] = dictionary[b'data'].astype(np.float64)
            dictionary[b'data'] /= 255
            dictionary[b'data'] -= 0.5
            labels = np.zeros((len(dictionary[b'labels']), 10))
            for i in range(len(dictionary[b'labels'])):
                labels[i, dictionary[b'labels'][i]] = 1
            dictionary[b'labels'] = labels
        return dictionary[b'data'], dictionary[b'labels']

    @staticmethod
    def show_img_array(flat, normalized=True):
        if normalized:
            flat += 0.5
            flat *= 255

        img_array = np.zeros((32, 32, 3))
        for channel in range(3):
            for row in range(32):
                for collum in range(32):
                    img_array[row, collum, channel] = flat[channel * 1024 + row * 32 + collum]
        with Image.fromarray(img_array.astype(np.uint8)) as img:
            img.show()
