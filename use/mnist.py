from keras.datasets import mnist
import numpy as np


class Mnist(object):

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def getX(self, n=10000, flat=True):
        """

        :param n: amount of samples
        :param flat: boolean flag. True to represent samples as 1D matrix of size 784
        False to 2D matrix of size 28x28
        :return:
        """
        XXX = (list(self.x_train) + list(self.x_test))[:n]
        x = []
        for sample in XXX:
            if flat:
                x.append(np.array(sample).flatten())
            else:
                x.append(np.array(sample))

        return np.array(x)

    def getY(self, n=10000):
        return np.array((list(self.y_train) + list(self.y_test))[:n])