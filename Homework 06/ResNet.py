import tensorflow as tf

from ResidualBlock import *

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), activation="relu", padding='valid'),
            
            ResidualBlock(n_filters= 32, out_filters=32),
            ResidualBlock(n_filters= 16, out_filters=16),
            ResidualBlock(n_filters= 8, out_filters=8),


            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]
        
    #@tf.function
    def call(self, inputs, train):
        
        x = inputs

        for i, layer in enumerate(self.layer_list):

            if i in [1,2,3]:
                x = layer(x, train)
            else:
                x = layer(x)
        
        return x

    