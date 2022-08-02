import numpy as np


def sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(-a * (x + b)))


def unit_norm(x):
    return x / np.sqrt(np.sum(x**2, axis=1))[:, np.newaxis]
