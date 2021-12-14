import tensorflow as tf

class Encoder(tf.keras.layers.Layer): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        pass
    
    @tf.function
    def call(self, x):
        pass