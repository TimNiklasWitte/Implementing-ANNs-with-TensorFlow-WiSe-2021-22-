import tensorflow as tf

class Discriminator(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_list = [
                    
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
        
            tf.keras.layers.Flatten(),

            #tf.keras.layers.Dense(50, activation='sigmoid'),

            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]

    @tf.function
    def call(self, x, training=False):

        for layer in self.layer_list:
            x = layer(x) 
            # if isinstance(layer, tf.keras.layers.Flatten):
            #     x = layer(x)
            # else:
            #     x = layer(x) 
            # try:
            #     x = layer(x,training)
            # except:
            #     x = layer(x) 
        return x