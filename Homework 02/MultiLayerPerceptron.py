from PerceptronLayer import PerceptronLayer
from Util import *

class MultiLayerPerceptron:

    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            layer = PerceptronLayer(sizes[i], sizes[i+1])
            self.layers.append(layer)

    def forward_step(self, x):
        for layer in self.layers:
            x = layer.activate(x)
        return x

    def backprop_step(self, x, t, epsilon):

        # Forward step saving weights
        activations = [x]
        for layer in self.layers:
            x = activations[-1]
            y = layer.activate(x)
            activations.append(y)

        # Initial delta
        outputLayer = self.layers[-1]
        y = activations[-1]
        delta = (t - y) * (y * (1 - y))
        outputLayer.adapt(activations[-2], delta, epsilon)

        # Propagation error backwards
        layers_withoutOutput = self.layers[:-1]
        for i in range(len(layers_withoutOutput)):

            # Reverse
            N = len(layers_withoutOutput) - i - 1

            # Calc delta
            weights_withoutBias = self.layers[N + 1].getWeightMatrix()[:, :-1] # ignore Bias
            sigmoidPrime = activations[N + 1] * (np.ones_like(activations[N + 1]) - activations[N + 1])
            delta = (delta.T @ weights_withoutBias) * sigmoidPrime

            self.layers[N].adapt(activations[N], delta, epsilon)
