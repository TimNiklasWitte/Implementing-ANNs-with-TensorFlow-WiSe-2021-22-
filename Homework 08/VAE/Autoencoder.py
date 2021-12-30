import tensorflow as tf
import numpy as np

from Encoder import * 
from Decoder import *

class Autoencoder(tf.keras.Model):
    def __init__(self, embedding_size):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(embedding_size)
        self.decoder = Decoder()

    @tf.function
    def call(self, x):
        embedding, mu, sigma = self.encoder(x)
        decoded = self.decoder(embedding)
        return decoded, mu, sigma

    
    @tf.function
    def train_step(self, input, target, reconstruction_loss, optimizer):
        # loss_object and optimizer_object are instances of respective tensorflow classes
        with tf.GradientTape() as tape:
            prediction, mu, sigma = self(input)
            loss = self.my_loss_function(target, prediction, reconstruction_loss, mu, sigma)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def test(self, test_data, reconstruction_loss):
        # test over complete test data
        test_loss_aggregator = []
        for input, target, _ in test_data: # ignore label            
            prediction, mu, sigma = self(input)   
            sample_test_loss = self.my_loss_function(target, prediction, reconstruction_loss, mu, sigma)
            test_loss_aggregator.append(sample_test_loss.numpy())

        test_loss = tf.reduce_mean(test_loss_aggregator)
        return test_loss
    
    @tf.function
    def my_loss_function(self, input, target, reconstruction_loss, mu, sigma):

        recon_loss = reconstruction_loss(input, target)
        
        # KL divergence
        kl_loss = -5e-4 * tf.keras.backend.mean(1 + sigma - tf.keras.backend.square(mu) - tf.keras.backend.exp(sigma), axis=-1)
        return tf.keras.backend.mean(recon_loss + kl_loss)