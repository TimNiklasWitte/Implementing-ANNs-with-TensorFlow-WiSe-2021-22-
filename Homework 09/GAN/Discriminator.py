import tensorflow as tf

class Discriminator(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(Discriminator, self).__init__()

        dropout_amount = 0.1

        self.layer_list = [
            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid'),
     
        ]

    @tf.function
    def call(self, x, training=False):

        for layer in self.layer_list: 
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training)
            else:
                x = layer(x) 
        return x