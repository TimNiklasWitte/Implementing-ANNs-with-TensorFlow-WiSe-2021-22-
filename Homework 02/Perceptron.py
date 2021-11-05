import numpy as np
from Util import *

import numpy as np

class Perceptron:

    def __init__(self, dim_in):
        self.weights = np.random.normal(size=dim_in + 1)

    def activate(self, x):
        x = np.append(x, 1) # bias
        self.drive = self.weights @ x
        return sigmoid(self.drive)

    def adapt(self, x, delta, epsilon):
        x = np.append(x, 1) # bias
        self.weights += epsilon * delta * x
