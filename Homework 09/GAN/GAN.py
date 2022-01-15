from ast import Global
import tensorflow as tf

from Generator import *
from Discriminator import *

import numpy as np

class GAN(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(GAN, self).__init__()

        self.G = Generator()
        self.D = Discriminator()

        self.D_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.G_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.batch_size = 256
        self.noise_dim = 100
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(self, real_images):

        
        batch_size = real_images.shape[0]
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = self.G(noise, training=True)

            real_data_pred = self.D(real_images, training=True)
            fake_data_pred = self.D(fake_images, training=True)

            g_loss = self.generator_loss(fake_data_pred)
            d_loss = self.discriminator_loss(real_data_pred, fake_data_pred)

        g_gradients = g_tape.gradient(g_loss, self.G.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.D.trainable_variables)

        self.G_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(d_gradients, self.D.trainable_variables))

        return g_loss, d_loss 


    def test(self, test_data):
        # test over complete test data
        
        test_accuracy_aggregator_fake = []
        test_accuracy_aggregator_real = []
        test_loss_aggregator_g = []
        test_loss_aggregator_d = []

        for real_images in test_data:
            batch_size = real_images.shape[0]
            noise = tf.random.normal(shape=(batch_size, self.noise_dim))

            fake_images = self.G(noise, training=True)

            real_data_pred = self.D(real_images, training=True)
            fake_data_pred = self.D(fake_images, training=True)

            g_loss = self.generator_loss(fake_data_pred)
            d_loss = self.discriminator_loss(real_data_pred, fake_data_pred)
            
            test_loss_aggregator_g.append(g_loss)
            test_loss_aggregator_d.append(d_loss)

                
            sample_test_accuracy_fake =  np.zeros_like(fake_data_pred) == np.round(fake_data_pred, 1) # <- Binary classification
            sample_test_accuracy_real =  np.ones_like(real_data_pred) == np.round(real_data_pred, 1)

            test_accuracy_aggregator_fake.append(np.mean(sample_test_accuracy_fake))
            test_accuracy_aggregator_real.append(np.mean(sample_test_accuracy_real))

        test_loss_g = np.mean(test_loss_aggregator_g)
        test_loss_d = np.mean(test_loss_aggregator_d)
        test_accuracy_fake = np.mean(test_accuracy_aggregator_fake)
        test_accuracy_real = np.mean(test_accuracy_aggregator_real)

        return test_loss_g, test_loss_d, test_accuracy_real, test_accuracy_fake


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


