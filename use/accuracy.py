from munkres import Munkres
import numpy as np


def accuracy(true_labels, labels):
    """
    Calculate accuracy of unsupervised clustering using Hungarian algorithm
    :param true_labels:
    :param labels:
    :return:
    """
    n = np.unique(labels).size
    matrix = np.zeros((n, n))

    for true, predict in zip(true_labels, labels):
        for i in range(n):
            if true == i:
                matrix[predict][i] += 1
    cost_matrix = matrix * (-1)
    m = Munkres()
    indexes = m.compute(cost_matrix)

    n_samples = labels.size
    n_true_predicted = sum([matrix[ind[0]][ind[1]] for ind in indexes])

    return n_true_predicted / n_samples