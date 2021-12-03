import tensorflow as tf

from ResNet.ResidualBlock import *

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=55, kernel_size=(3,3), strides=(1,1), padding='valid'),
            
            ResidualBlock(n_filters=60, out_filters=65, mode="normal"),
            ResidualBlock(n_filters=70, out_filters=65, mode="strided"),

            ResidualBlock(n_filters=70, out_filters=75, mode="normal"),
            ResidualBlock(n_filters=80, out_filters=75, mode="strided"),
            
            ResidualBlock(n_filters=80, out_filters=85, mode="normal"),
            ResidualBlock(n_filters=90, out_filters=85, mode="strided"), # <- remove?

            #ResidualBlock(n_filters=45, out_filters=50, mode="normal"),
            #ResidualBlock(n_filters=55, out_filters=50, mode="strided"),

            #ResidualBlock(n_filters=50, out_filters=50, mode="constant"),

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

    