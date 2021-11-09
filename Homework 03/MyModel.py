from tensorflow.keras.layers import Dense
import tensorflow as tf

import numpy as np
from SimpleDense import SimpleDense 

class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = SimpleDense(256, activation=tf.nn.sigmoid)
        self.dense2 = SimpleDense(256, activation=tf.nn.sigmoid) 
        self.out = SimpleDense(10, activation=tf.nn.softmax) 

    @tf.function
    def call(self, inputs):
        
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x

    @tf.function 
    def train_step(self, input, target, loss_function, optimizer):
            # loss_object and optimizer_object are instances of respective tensorflow classes
            with tf.GradientTape() as tape:
                prediction = self(input)
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
            
            prediction = self(input)
            sample_test_loss = loss_function(target, prediction)
            sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
            sample_test_accuracy = np.mean(sample_test_accuracy)
            test_loss_aggregator.append(sample_test_loss.numpy())
            test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

        return test_loss, test_accuracy