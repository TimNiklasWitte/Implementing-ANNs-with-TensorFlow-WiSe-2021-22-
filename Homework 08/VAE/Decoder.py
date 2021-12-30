import tensorflow as tf

class Decoder(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):

        super(Decoder, self).__init__()
        self.layer_list = [
            
            tf.keras.layers.Dense(7*7*16),

            tf.keras.layers.Reshape((7,7,16)), 

            tf.keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=(3,3), strides=2, padding='same', activation='relu'),

            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='tanh')
        ]

    @tf.function
    def call(self, x):
        
        for layer in self.layer_list:
            x = layer(x)
        return x