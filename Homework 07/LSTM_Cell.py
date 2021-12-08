import tensorflow as tf

class LSTM_Cell:

    # units = hidden size
    def __init__(self,units):
        self.dense_layer_hstate = tf.keras.layers.Dense(units, activation=None, use_bias=False)