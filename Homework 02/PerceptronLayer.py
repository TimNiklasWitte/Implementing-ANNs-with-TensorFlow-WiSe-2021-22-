from Perceptron import Perceptron

import numpy as np

class PerceptronLayer:

    def __init__(self, dim_in, dim_out):
        self.perceptrons = [Perceptron(dim_in) for _ in range(dim_out)]

    def activate(self, x):
        activation = np.array([p.activate(x) for p in self.perceptrons])
        return activation

    def adapt(self, x, deltas, epsilon):
        for perceptron, delta in zip(self.perceptrons, deltas):
            perceptron.adapt(x, delta, epsilon)

    def getWeightMatrix(self):
        return np.array([p.weights for p in self.perceptrons])
