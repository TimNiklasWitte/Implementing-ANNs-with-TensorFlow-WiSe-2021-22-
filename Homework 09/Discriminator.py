import tensorflow as tf

class Discriminator(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_list = [
                    
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
        
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(100, activation='sigmoid'),

            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]

    def call(self, x, training=False):

        for layer in self.layer_list:
            try:
                x = layer(x,training)
            except:
                x = layer(x)  
        return x