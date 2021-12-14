import tensorflow as tf

class Encoder(tf.keras.layers.Layer): # <-- Needed to make parameters trainable and to be callable
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.layer_list = [
            #tf.keras.layers.Conv2D(15, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(8, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(embedding_size)
        ]
    
    #@tf.function
    def call(self, x):
        
        ashape = None
        for layer in self.layer_list:
            
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Conv2D):
                ashape = x.shape

        return x, ashape