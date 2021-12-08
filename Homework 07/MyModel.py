from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

from LSTM_Layer import *
from LSTM_Cell import *

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.layer_list = [

            LSTM_Layer(LSTM_Cell(3)),
            tf.keras.layers.Dense(1, activation="sigmoid")

        ]

    @tf.function
    def call(self, x):

        for layer in self.layer_list:
          
            
            if isinstance(layer, LSTM_Layer):
                batchSize = x.shape[0]
                states = layer.zero_states(batchSize)
                x = layer.call(x, states)

            else:
                x = x[:,:,2] # look only at output size
                x = layer(x)

        return x