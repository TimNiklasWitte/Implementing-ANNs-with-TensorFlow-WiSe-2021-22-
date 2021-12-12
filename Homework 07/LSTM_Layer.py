from LSTM_Cell import *

import numpy as np

class LSTM_Layer(tf.keras.layers.Layer): # <-- Needed to make parameters trainable and to be callable

    """
    The cell will be called n-times recurrent with the corresponding input for that time step.
    n is time length of the time step
    """
    def __init__(self, cell):
        super(LSTM_Layer, self).__init__()
        self.cell = cell
    
    @tf.function
    def call(self, x, states):

        """
        Propagate the input towards all time-steps -> recurrent

        Args:
            x input of the cell, shape: (batch size, seq len, input size)
            states inital state, tuple (hidden_state, cell_state)

        Return:
            Output of the layer: [batch size, seq len, output size]
        """

        seq_len = x.shape[1]

        # shape: batch, time-steps, h_dim
        hidden_states = tf.TensorArray(dtype=tf.float32, size=seq_len)
        
        for t in tf.range(seq_len): 
            input_t = x[:,t,:]
            
            states  = self.cell.call(input_t, states)

            # Extract hidden state (output)
            # cell_state not needed, this state is not part of the output!
            hidden_state, _ = states 
            hidden_states = hidden_states.write(t, hidden_state) 

        # transpose the sequence of hidden_states from TensorArray accordingly (batch and time dimensions switched)
        return tf.transpose(hidden_states.stack(), [1,0,2])

    def zero_states(self, batch_size):

        """
        Create an init state (all zeros) for hidden_state and cell_state
            -> Return a two tuple. Each entry is a Tensor with zeros
        (note: hidden_size = number of units = length of vector of hidden state)

        Args:
            batch_size

        Return:
            Tuple with Zero Tensors: ( [batch_size, hidden_size], batch_size, hidden_size] )
        """

        return (tf.zeros(shape=(batch_size, self.cell.units)), tf.zeros(shape=(batch_size, self.cell.units)))
