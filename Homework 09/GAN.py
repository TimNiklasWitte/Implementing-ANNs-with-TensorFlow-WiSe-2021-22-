import tensorflow as tf

from Generator import *
from Discriminator import *

import numpy as np

class GAN(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(GAN, self).__init__()

        self.G = Generator()
        self.D = Discriminator()

        learning_rate = 0.01
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.loss_function = tf.keras.losses.BinaryCrossentropy()


    @tf.function
    def train_step(self, data):

        z = tf.random.normal([data.shape[0],100])

    
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            fake_data = self.G(z)
            fake_data_pred = self.D(fake_data)
            real_data_pred = self.D(data)
                
            #D_loss = -tf.math.reduce_mean( tf.math.log(real_data_pred) + tf.math.log(1-fake_data_pred) )
            #G_loss = tf.math.reduce_mean( tf.math.log(1-fake_data_pred) )

            D_loss_fake = self.loss_function(fake_data_pred, tf.zeros_like(fake_data_pred))
            D_loss_real = self.loss_function(real_data_pred, tf.ones_like(real_data_pred))
            D_loss = D_loss_fake + D_loss_real
            D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)
            
            real self.G(z)
            D_loss_real = self.loss_function(real_data_pred, tf.ones_like(real_data_pred))
            G_loss = D_loss_fake + D_loss_real
            G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)
        
        
        self.D_optimizer.apply_gradients(zip(D_gradients, self.D.trainable_variables))
        self.G_optimizer.apply_gradients(zip(G_gradients, self.G.trainable_variables))
        
        return G_loss


    def test(self, test_data):
        # test over complete test data
        
        test_accuracy_aggregator_fake = []
        test_accuracy_aggregator_real = []
        test_loss_aggregator = []

        for data in test_data:
            
            z = tf.random.normal([data.shape[0],100])

            fake_data = self.G(z)
            fake_data_pred = self.D(fake_data)
            real_data_pred = self.D(data)

            D_loss_fake = self.loss_function(fake_data_pred, tf.zeros_like(fake_data_pred))
            D_loss_real = self.loss_function(real_data_pred, tf.ones_like(real_data_pred))
            G_loss = D_loss_fake + D_loss_real

                    
            sample_test_accuracy_fake =  np.zeros_like(fake_data_pred) == np.round(fake_data_pred, 1) # <- Binary classification
            sample_test_accuracy_real =  np.ones_like(real_data_pred) == np.round(real_data_pred, 1)

            sample_test_accuracy_fake = np.mean(sample_test_accuracy_fake)
            sample_test_accuracy_real = np.mean(sample_test_accuracy_real)

            test_loss_aggregator.append(G_loss.numpy())

            test_accuracy_aggregator_fake.append(np.mean(sample_test_accuracy_fake))
            test_accuracy_aggregator_real.append(np.mean(sample_test_accuracy_real))

        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy_fake = tf.reduce_mean(test_accuracy_aggregator_fake)
        test_accuracy_real = tf.reduce_mean(test_accuracy_aggregator_real)

        print(test_accuracy_fake)
        print(test_accuracy_real)
        accuracy = (test_accuracy_fake + test_accuracy_real)/2
        return test_loss, accuracy