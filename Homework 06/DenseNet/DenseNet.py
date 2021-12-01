import tensorflow as tf

from DenseNet.DenseBlock import *
from DenseNet.TransitionLayer import *

class DenseNet(tf.keras.Model):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.layer_list = [
            # Initial Conv layer before the first res block
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),

            DenseBlock(n_filters=128, new_channels=32),
            DenseBlock(n_filters=256, new_channels=64),
            DenseBlock(n_filters=128, new_channels=32),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(10, activation=tf.nn.softmax) 
        ]

    def call(self, inputs, train):
        x = inputs

        for layer in self.layer_list:

            if isinstance(layer, DenseBlock):
                x = layer(x, train)
            else:
                x = layer(x)
        
        return x