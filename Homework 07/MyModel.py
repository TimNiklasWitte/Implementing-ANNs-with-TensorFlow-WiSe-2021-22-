from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

from LSTM_Layer import *
from LSTM_Cell import *

class MyModel(tf.keras.Model):
    """
    General idea
    (1) Dense Layer -> Increase input from 5 to 32 
    (2) LSTM_Layer 
    (3) Sigmoid -> Classification: '1': Output > 0.5, otherwise: '0'
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # batch_size = 32
        # seq_len = 25
        # input_size = 1 (only a number)
        self.layer_list = [

            # input shape: (batch_size, seq_len, input_size) = (32, 25, 1) 
            # (1)
            tf.keras.layers.Dense(32, activation="tanh"), # output shape: 32, 25, 32 
            # (2)
            LSTM_Layer(LSTM_Cell(100)), # output shape: 32, 25, 100   (all time steps are returned)
            # (3)
            tf.keras.layers.Dense(1, activation="sigmoid") # output shape: 32, 25, 1 
                                                           # call function returns only last time step 
        ]
      
    @tf.function
    def call(self, x):
        
        """
        Propagate the input towards all layers
        Args:
            x input

        Return:
            output of the model, considers last time step
        """

        batchSize = x.shape[0]
  
        for layer in self.layer_list:
            
            if isinstance(layer, LSTM_Layer):
                # Propagate states (hidden_state, cell_state)
                states = layer.zero_states(batchSize)
                x = layer(x, states)

            else:
                x = layer(x)

        # Only last time step 
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

        # prediction shape: (batch_size, 1)
        # target shape: (batch_size, ) => Add dim for comparing them
        target = np.expand_dims(target, -1) # now: (batch_size, 1)
        
        sample_test_accuracy =  np.round(target, 1) == np.round(prediction, 1) # <- Binary classification
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    return test_loss, test_accuracy