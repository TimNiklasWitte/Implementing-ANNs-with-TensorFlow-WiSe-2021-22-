import tensorflow as tf

class Discriminator(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(Discriminator, self).__init__()

        dropout_amount = 0.1

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation=None, kernel_initializer=tf.keras.initializers.glorot_uniform),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            #tf.keras.layers.Dropout(dropout_amount),

            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation=None, kernel_initializer=tf.keras.initializers.glorot_uniform),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            #tf.keras.layers.Dropout(dropout_amount),
                
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=None, kernel_initializer=tf.keras.initializers.glorot_uniform),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            #tf.keras.layers.Dropout(dropout_amount),

            # tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation=None, kernel_initializer=tf.keras.initializers.glorot_uniform),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Activation(tf.nn.relu),
            #tf.keras.layers.Dropout(dropout_amount),
                
            tf.keras.layers.GlobalAvgPool2D(),

            tf.keras.layers.Dense(units=1, activation = 'sigmoid')
        ]

    @tf.function
    def call(self, x, training=False):

        for layer in self.layer_list:
         
            if isinstance(layer, tf.keras.layers.Dropout) or isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training)
            else:
                x = layer(x) 
        return x