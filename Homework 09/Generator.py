import tensorflow as tf

class Generator(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(Generator, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(units=7*7*256, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(7,7,256)),

            tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding='same', activation=None, kernel_initializer=tf.keras.initializers.glorot_uniform),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation=None, kernel_initializer=tf.keras.initializers.glorot_uniform),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding='same', activation=None, kernel_initializer=tf.keras.initializers.glorot_uniform),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same', activation='tanh', kernel_initializer=tf.keras.initializers.glorot_uniform)
        ]

    @tf.function
    def call(self, x, training=False):

        for layer in self.layer_list:
         
            if isinstance(layer, tf.keras.layers.Dropout) or isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training)
            else:
                x = layer(x) 
        return x