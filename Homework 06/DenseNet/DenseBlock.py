import tensorflow as tf

class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, n_filters, new_channels):
        
        super(DenseBlock, self).__init__()

        self.layer_list = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(n_filters, kernel_size=(1,1), padding="valid"),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(new_channels, kernel_size=(3,3), padding="same"),
        ]

    
    def call(self, x, train):
        
        input = x

        # propagte towards all layers
        for layer in self.layer_list:
            
            # No BatchNormalization during training
            if not train:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    continue

            x = layer(x)

        return tf.keras.layers.Concatenate(axis=-1)([input, x]) 

    