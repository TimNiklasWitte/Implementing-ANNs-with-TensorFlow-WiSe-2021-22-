import tensorflow as tf
import numpy as np

from ResNet.ResidualBlock import *

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=55, kernel_size=(3,3), strides=(1,1), padding='valid'),
            
            ResidualBlock(n_filters=60, out_filters=65, mode="normal"),
            ResidualBlock(n_filters=70, out_filters=65, mode="strided"),

            ResidualBlock(n_filters=70, out_filters=75, mode="normal"),
            ResidualBlock(n_filters=80, out_filters=75, mode="strided"),
            
            ResidualBlock(n_filters=80, out_filters=85, mode="normal"),
            ResidualBlock(n_filters=90, out_filters=85, mode="strided"), 

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]
        
    @tf.function
    def call(self, inputs, train):
        
        x = inputs

        for layer in self.layer_list:

            if isinstance(layer, ResidualBlock):
                x = layer(x, train)
            else:
                x = layer(x)
        
        return x

    
    @tf.function 
    def train_step(self, input, target, loss_function, optimizer):
        # loss_object and optimizer_object are instances of respective tensorflow classes
        with tf.GradientTape() as tape:
            prediction = self(input, True)
            loss = loss_function(target, prediction)
            gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    # @tf.function # <-- error?
    def test(self, test_data, loss_function):
        # test over complete test data

        test_accuracy_aggregator = []
        test_loss_aggregator = []

        for (input, target) in test_data:
                
            prediction = self(input, False)
            sample_test_loss = loss_function(target, prediction)
            sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
            sample_test_accuracy = np.mean(sample_test_accuracy)
            test_loss_aggregator.append(sample_test_loss.numpy())
            test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

        return test_loss, test_accuracy

    