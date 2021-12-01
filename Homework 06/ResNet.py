import tensorflow as tf

from ResidualBlock import *

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), activation="relu", padding='valid'),
            
            ResidualBlock(n_filters=40, out_filters=64, mode="normal"),
            ResidualBlock(n_filters=128, out_filters=150, mode="normal"),

            ResidualBlock(n_filters=100, out_filters=150, mode="strided"),
            ResidualBlock(n_filters=100, out_filters=150, mode="strided"),
            ResidualBlock(n_filters=100, out_filters=150, mode="strided"),

            ResidualBlock(n_filters=200, out_filters=250, mode="normal"),

            ResidualBlock(n_filters=250, out_filters=250, mode="constant"),
            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]
        
    #@tf.function
    def call(self, inputs, train):
        
        x = inputs

        for layer in self.layer_list:

            if isinstance(layer, ResidualBlock):
                x = layer(x, train)
            else:
                x = layer(x)
        
        return x

    