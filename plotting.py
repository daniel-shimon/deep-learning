import matplotlib.pyplot as plt
import random

graphs = []


def set_ylabel(label):
    plt.ylabel(label)


def replot():
    plt.clf()
    for graph in graphs:
        args, kwargs = graph.get_plot_args()
        plt.plot(*args, **kwargs)
    plt.legend()
    plt.pause(0.01)


class Graph(object):
    def __init__(self, label='', points=1000):
        self.label = label
        self.points = points
        graphs.append(self)
        self.ys = []
        self.xs = []

    def maybe_add(self, i, x, y, plot=False):
        if len(self.xs) == self.points:
            r = random.randint(0 + 1, i)
            if r < self.points:
                self.pop(r)
                self.add(x, y)
                self.maybe_plot(plot)
        else:
            self.add(x, y)
            self.maybe_plot(plot)

    def pop(self, i):
        self.xs.pop(i)
        self.ys.pop(i)

    def add(self, x, y):
        self.xs.append(x)
        self.ys.append(y)

    def get_plot_args(self):
        return (self.xs, self.ys), {'label': self.label}

    def clear(self, plot=False):
        self.xs.clear()
        self.ys.clear()
        self.maybe_plot(plot)

    @staticmethod
    def maybe_plot(plot):
        if plot:
            replot()

plt.ion()
