import tensorflow as tf

from ResNet.ResidualBlock import *

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), activation="relu", padding='valid'),
            
            # Change number of channels
            ResidualBlock(n_filters=64, out_filters=128, mode="normal"),

            # Shrinks feature maps, changes n of channels 
            ResidualBlock(n_filters=64, out_filters=128, mode="strided"),

            # Keeps feature map size and n of channels
            ResidualBlock(n_filters=128, out_filters=128, mode="constant"),

            # Change number of channels
            ResidualBlock(n_filters=128, out_filters=256, mode="normal"),

            # Shrinks feature maps, changes n of channels 
            ResidualBlock(n_filters=128, out_filters=256, mode="strided"),

            # Keeps feature map size and n of channels
            ResidualBlock(n_filters=256, out_filters=256, mode="constant"),

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

    