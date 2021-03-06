from tensorflow.keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.layer_list = [

            # Increase number of filters after first MaxPool -> Performance gain
            # padding='valid' -> Reduces output size
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), activation="relu", padding='valid'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), activation="relu", padding='valid'),

            # Pooling: pool size = stride size
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(3,3), padding='same'), 
            
            # Now: Increase number of filters gradually per layer
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation="relu", padding='valid'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation="relu", padding='valid'),

            # Vectorize = Flatten by taken avg -> Sum up all 128 resulting matrices (filters) from previous layer to 128 values
            # Each matrix (filter) is reduced to a single value (average)
            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(10, activation=tf.nn.softmax)

        ]

    @tf.function
    def call(self, inputs):

        x = inputs

        for layer in self.layer_list:
            x = layer(x)

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
    sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy
