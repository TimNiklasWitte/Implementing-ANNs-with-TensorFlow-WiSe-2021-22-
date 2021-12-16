import tensorflow as tf
import numpy as np

from ConvolutionalAutoencoder.Encoder import * 
from ConvolutionalAutoencoder.Decoder import *

class Autoencoder(tf.keras.Model):
    def __init__(self, x, embedding_size):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(embedding_size)
        shapeAfterLastConv, denseLayerSize = self.encoder.getShapes(x)
        self.decoder = Decoder(shapeAfterLastConv, denseLayerSize)

    @tf.function
    def call(self, x):
        embedding = self.encoder(x)
        decoded = self.decoder(embedding)
        return decoded

    
    @tf.function
    def train_step(self, input, target, loss_function, optimizer):
        # loss_object and optimizer_object are instances of respective tensorflow classes
        with tf.GradientTape() as tape:
            prediction = self(input)
            loss = loss_function(target, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def test(self, test_data, loss_function):
        # test over complete test data
        test_loss_aggregator = []
        for input, target, _ in test_data: # ignore label            
            prediction = self(input)   
            sample_test_loss = loss_function(target, prediction)
            test_loss_aggregator.append(sample_test_loss.numpy())

        test_loss = tf.reduce_mean(test_loss_aggregator)
        return test_loss