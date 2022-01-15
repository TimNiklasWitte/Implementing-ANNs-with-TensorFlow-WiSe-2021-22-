from ast import Global
import tensorflow as tf

from Generator import *
from Critic import *

import numpy as np

class WGAN(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(WGAN, self).__init__()

        self.G = Generator()
        self.C = Critic()

        self.G_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.C_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.batch_size = 256
        self.noise_dim = 100
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(self, real_images):

        
        batch_size = real_images.shape[0]
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))

        with tf.GradientTape() as g_tape, tf.GradientTape() as c_tape:
            fake_images = self.G(noise, training=True)

            real_data_output = self.C(real_images, training=True)
            fake_data_output = self.C(fake_images, training=True)

            g_loss = self.generator_loss(fake_data_output)
            c_loss = self.critic_loss(real_data_output, fake_data_output)

            g_gradients = g_tape.gradient(g_loss, self.G.trainable_variables)
            c_gradients = c_tape.gradient(c_loss, self.C.trainable_variables)

        self.G_optimizer.apply_gradients(zip(g_gradients, self.G.trainable_variables))
        self.C_optimizer.apply_gradients(zip(c_gradients, self.C.trainable_variables))

        return tf.reduce_mean(g_loss), tf.reduce_mean(c_loss)


    def test(self, test_data):
        # test over complete test data
        
        test_loss_aggregator_g = []
        test_loss_aggregator_c = []

        for real_images in test_data:
            batch_size = real_images.shape[0]
            noise = tf.random.normal(shape=(batch_size, self.noise_dim))

            fake_images = self.G(noise, training=True)

            real_data_output = self.C(real_images, training=True)
            fake_data_output = self.C(fake_images, training=True)

            g_loss = self.generator_loss(fake_data_output)
            d_loss = self.critic_loss(real_data_output, fake_data_output)
            
            test_loss_aggregator_g.append(g_loss)
            test_loss_aggregator_c.append(d_loss)

        test_loss_g = np.mean(test_loss_aggregator_g)
        test_loss_c = np.mean(test_loss_aggregator_c)

        return test_loss_g, test_loss_c 


    def critic_loss(self, real_output, fake_output):
        # D(x) - D(G(z))
        return real_output - fake_output

    def generator_loss(self, fake_output):
        # D(G(z))
        return fake_output


