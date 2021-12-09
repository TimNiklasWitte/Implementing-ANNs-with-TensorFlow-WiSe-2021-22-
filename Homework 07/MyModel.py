from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

from LSTM_Layer import *
from LSTM_Cell import *

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(32, activation="sigmoid"),
            tf.keras.layers.Dense(16, activation="sigmoid"),
            LSTM_Layer(LSTM_Cell(10)),
        

            tf.keras.layers.Dense(1, activation="sigmoid")
        ]
      
    @tf.function
    def call(self, x):
        batchSize = x.shape[0]
  

        for layer in self.layer_list:
            
            if isinstance(layer, LSTM_Layer):
                
                states = layer.zero_states(batchSize)
                x = layer(x, states)

            else:
                x = layer(x)

        
        return x[:,-1,:]


@tf.function
def train_step(model, input, target, loss_function, optimizer):
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def test(model, test_data, loss_function):
    # test over complete test data
    
    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        
        prediction = model(input)
        
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy =  np.round(target, 1) == np.round(prediction, 1) # <- Binary classification
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    return test_loss, test_accuracy