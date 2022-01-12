import tensorflow as tf

class Generator(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(Generator, self).__init__()

        self.layer_list = [
            tf.keras.Input((100,1)),

            tf.keras.layers.Reshape((10,10,1)), 

            tf.keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=(3,3), strides=2, padding='same', activation='relu'),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(28*28*1, activation='tanh'), # make reshape simpler
            tf.keras.layers.Reshape((28,28,1)), 

            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='tanh')
        ]
    
    def call(self, x, training=False):

        for layer in self.layer_list:
            try:
                x = layer(x,training)
            except:
                x = layer(x) 
        return x