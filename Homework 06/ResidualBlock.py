import tensorflow as tf

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, n_filters=64, out_filters=256):
        super(ResidualBlock, self).__init__()

        self.layer_list = [

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=n_filters, kernel_size =(1,1)),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=n_filters, kernel_size =(3,3), padding="same"),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=out_filters, kernel_size =(1,1))

        ]

    #def build(self, n_filters=64, out_filters=256):
        
        

        

    #@tf.function 
    def call(self, x, train):
        
        input = x

        # propagte towards all layers
        for i, layer in enumerate(self.layer_list):
            
            # No BatchNormalization during training
            if not train:
                if i % 3 == 0:
                    continue

            x = layer(x)
        
        return tf.keras.layers.Add()([x, input])  
