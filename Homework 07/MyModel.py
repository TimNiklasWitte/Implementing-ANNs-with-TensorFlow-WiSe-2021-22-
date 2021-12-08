from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

from LSTM_Layer import *
from LSTM_Cell import *

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.layer_list = [
            LSTM_Layer(LSTM_Cell(75)),
            LSTM_Layer(LSTM_Cell(50)),
            LSTM_Layer(LSTM_Cell(25)),
         
            tf.keras.layers.Dense(1, activation="sigmoid")
        ]

    @tf.function
    def call(self, x):

        denseBlock = False # mutiple dense layers behind LSTM_Layer
        for layer in self.layer_list:
          
            
            if isinstance(layer, LSTM_Layer):
                batchSize = x.shape[0]
                states = layer.zero_states(batchSize)
                x = layer.call(x, states)#[:,-1,:] # consider only last time testep

            else:
                
                if not denseBlock:
                    x = x[:,-1,:] # look only at output size

                denseBlock = True
                x = layer(x)

        return x


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