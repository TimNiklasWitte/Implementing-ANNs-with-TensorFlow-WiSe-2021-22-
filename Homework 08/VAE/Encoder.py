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

        self.mu_layer = tf.keras.layers.Dense(embedding_size)
        self.sigma_layer = tf.keras.layers.Dense(embedding_size)
    
    @tf.function
    def call(self, x):
        for layer in self.layer_list: 
            x = layer(x)

        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)

        eps = tf.keras.backend.random_normal(
                shape=(tf.keras.backend.shape(mu)[0], tf.keras.backend.shape(mu)[1])
                )  

        return mu + tf.keras.backend.exp(sigma/2) * eps, mu, sigma
    