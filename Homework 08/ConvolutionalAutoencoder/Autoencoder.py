import tensorflow as tf
import numpy as np

from ConvolutionalAutoencoder.Encoder import * 
from ConvolutionalAutoencoder.Decoder import *

class Autoencoder(tf.keras.Model):
    def __init__(self, embedding_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(embedding_size)
        self.decoder = Decoder()

    #@tf.function
    def call(self, x):
        embedding, shape = self.encoder(x)
        # print(embedding.shape)
        # print("----")
        # embedding = tf.keras.layers.Reshape( (-1,5, 3))(embedding)
        # print(embedding.shape)

        # embedding = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=(2,2), strides=2, padding='same', activation='relu')(embedding)
        # print(embedding.shape)

        decoded = self.decoder(embedding, shape)
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

        for (input, target) in test_data:

            prediction = self(input)        
            sample_test_loss = loss_function(target, prediction)

            test_loss_aggregator.append(sample_test_loss.numpy())
    
        test_loss = tf.reduce_mean(test_loss_aggregator)
        return test_loss