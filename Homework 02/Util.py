import numpy as np

def sigmoid(x):
    return 1/( 1 + np.exp(-x) )

def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)

def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    if x > 0:
        return 1
    else:
        return 0

def loss(t, y):
    return (t - y)**2
