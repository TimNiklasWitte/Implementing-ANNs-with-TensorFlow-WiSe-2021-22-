import tensorflow as tf

class Encoder(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.layer_list = [
                    
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
        
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(7*7*16),

            tf.keras.layers.Dense(embedding_size, activation='relu')
        ]
    
    @tf.function
    def call(self, x):
        for layer in self.layer_list: 
            x = layer(x)       
        return x
    
    #@tf.function
    def getShapes(self, x):
        
        shapeAfterLastConv = None
        denseLayerSize = None
        for layer in self.layer_list:
            
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Conv2D):
                shapeAfterLastConv = x.shape[1:] # ignore batch dim
            elif isinstance(layer, tf.keras.layers.Flatten):
                denseLayerSize = x.shape[1] # ignore batch dim

        return shapeAfterLastConv, denseLayerSize