import tensorflow as tf

class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, n_filters, new_channels):
        """
        -> Conv2D -> Conv2D

        -> BatchNormalization and Activation

        First Conv2D shall increase the number of filters
        Second Conv2D decrease the number of filters
        Overall, the number of filters shall be increased

        Args:
            n_filters Number of filters used in the 1st Conv2D
            new_channels Number of filters used in the 2nd Conv2D
        """
        super(DenseBlock, self).__init__()

        self.layer_list = [
            tf.keras.layers.BatchNormalization(epsilon=1.001e-05),
            tf.keras.layers.Activation(tf.nn.relu),

            # 1x1 conv -> No Padding ("valid") needed
            tf.keras.layers.Conv2D(n_filters, kernel_size=(1,1), padding="valid", use_bias=False),

            tf.keras.layers.BatchNormalization(epsilon=1.001e-05),
            tf.keras.layers.Activation(tf.nn.relu),

            # to be concatenated with the input
            tf.keras.layers.Conv2D(new_channels, kernel_size=(3,3), padding="same", use_bias=False),
        ]

    @tf.function
    def call(self, x, train):
        
        """
        Propagate the input towards all layers

        Args:
            x input
            train flag set if we train
        """

        input = x

        # Propagte towards all layers
        for layer in self.layer_list:
            
            # Replay train parameter to BatchNormalization layers during call
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training=train)
            else:
                x = layer(x)

        return tf.keras.layers.Concatenate(axis=-1)([input, x]) 

    

    