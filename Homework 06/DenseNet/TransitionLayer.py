import tensorflow as tf

class TransitionLayer(tf.keras.layers.Layer):

    def __init__(self, n_filters, pool_size=(2,2)):

        """
        Args:
            n_filter number of filters used in Conv2D
            pool_size pool size used in AvgPool2D
        """

        super(TransitionLayer, self).__init__()

        self.layer_list = [
            tf.keras.layers.BatchNormalization(epsilon=1.001e-05),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Conv2D(n_filters, kernel_size=(1,1), padding="valid", use_bias=False),

            tf.keras.layers.AvgPool2D(pool_size=pool_size, strides = (2,2), padding = 'valid')
        ]

    @tf.function
    def call(self, x, train):
        
        # propagte towards all layers
        for layer in self.layer_list:
            
            # Replay train parameter to BatchNormalization layers during call
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training=train)
            else:
                x = layer(x)

        return x