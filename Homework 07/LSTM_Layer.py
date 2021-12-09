from LSTM_Cell import *

import numpy as np

class LSTM_Layer(tf.keras.layers.Layer): # <-- Needed to make parameters trainable and to be callable

    def __init__(self, cell):
        super(LSTM_Layer, self).__init__()
        self.cell = cell
    
    @tf.function
    def call(self, x, states):

        # x shape = input shape = (batch size, seq len, input size)

        # batch, time-steps, h_dim
        all_states = tf.TensorArray(dtype=tf.float32, size=x.shape[1])
    
                
        for t in tf.range(x.shape[1]): # seq len
            input_t = x[:,t,:]
            states = self.cell.call(input_t, states)
            all_states = all_states.write(t, states[0]) # hidden state

        # output [batch size, seq len, output size]
        # transpose the sequence of hidden_states from TensorArray accordingly (batch and time dimensions switched)
        return tf.transpose(all_states.stack(), [1,0,2])

    def zero_states(self, batch_size):
        return (tf.zeros(shape=(batch_size, self.cell.units)), tf.zeros(shape=(batch_size, self.cell.units)))
