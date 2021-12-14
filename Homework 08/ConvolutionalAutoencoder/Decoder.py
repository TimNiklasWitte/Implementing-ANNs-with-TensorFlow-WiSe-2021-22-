import tensorflow as tf

class Decoder(tf.keras.layers.Layer): # <-- Needed to make parameters trainable and to be callable
    def __init__(self, shapeAfterLastConv, denseLayerSize):

        super(Decoder, self).__init__()
        self.layer_list = [
            
            tf.keras.layers.Dense(denseLayerSize, activation='tanh'),

            tf.keras.layers.Reshape(shapeAfterLastConv), 


            tf.keras.layers.Conv2DTranspose(16, kernel_size=(3,3), strides=2, padding='same'),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=(3,3), strides=2, padding='same'),
   
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='sigmoid')
        ]
    
    @tf.function
    def call(self, x):
        
        for layer in self.layer_list:
            x = layer(x)
        return x