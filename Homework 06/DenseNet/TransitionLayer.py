import tensorflow as tf

class TransitionLayer(tf.keras.layers.Layer):

    def __init__(self, n_filters, pool_size=(2,2)):
        super(TransitionLayer, self).__init__()

        self.layer_list = [

            tf.keras.layers.Conv2D(n_filters, kernel_size=(1,1), padding="valid"),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.AveragePooling2D(pool_size)

        ]

    def call(self, x, train):
        
        # propagte towards all layers
        for layer in self.layer_list:
            
            # No BatchNormalization during training
            if not train:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    continue

            x = layer(x)

        return x