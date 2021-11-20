from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
from keras.regularizers import l1
from keras.regularizers import l2

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x


class MyModel_Regularization_L1(tf.keras.Model):
    def __init__(self):
        super(MyModel_Regularization_L1, self).__init__()
        factor = 0.01
        self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=l1(factor), bias_regularizer=l1(factor))
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=l1(factor), bias_regularizer=l1(factor))
        self.out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer=l1(factor), bias_regularizer=l1(factor))

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x


class MyModel_Regularization_L2(tf.keras.Model):
    def __init__(self):
        super(MyModel_Regularization_L2, self).__init__()
        factor = 0.01
        self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=l2(factor), bias_regularizer=l2(factor))
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=l2(factor), bias_regularizer=l2(factor))
        self.out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer=l2(factor), bias_regularizer=l2(factor))

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x


class MyModel_Dropout(tf.keras.Model):
    def __init__(self):
        super(MyModel_Dropout, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)


    @tf.function
    def call(self, inputs):

        drop_rate = 0.1
        x = self.dense1(inputs)
        x = tf.nn.dropout(x, drop_rate)

        x = self.dense2(x)
        x = tf.nn.dropout(x, drop_rate)

        x = self.out(x)
        x = tf.nn.dropout(x, drop_rate)

        return x


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
    sample_test_accuracy =  np.round(target, 1) == np.round(prediction, 1)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy
