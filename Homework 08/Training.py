import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from ConvolutionalAutoencoder.Autoencoder import * 

import matplotlib.pyplot as plt

def main():

    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    train_dataset = train_ds.apply(prepare_mnist_data)
    test_dataset = test_ds.apply(prepare_mnist_data)

    train_dataset = train_dataset.take(1000)
    test_dataset = test_dataset.take(100)

     ### Hyperparameters
    num_epochs = 10
    learning_rate = 0.01

    # Initialize the model.
    model = Autoencoder(15)

    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    mse = tf.keras.losses.MeanSquaredError()

    # Initialize the optimizer: 
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []

    test_losses = []


    #testing once before we begin
    test_loss = model.test(test_dataset, mse)
    test_losses.append(test_loss)

    #check how model performs on train data once before we begin
    train_loss = model.test(train_dataset, mse)
    train_losses.append(train_loss)

    model.summary()

    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_losses[-1]}')

        #training (and checking in with training)
        epoch_loss_agg = []
        for input,target in train_dataset:
            train_loss = model.train_step(input, target, mse, optimizer)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        # testing, so we can track accuracy and test loss
        test_loss = model.test(test_dataset, mse)
        test_losses.append(test_loss)


    # for elem in train_dataset.take(1):
        
    #     orginal, noised = elem

    #     orginal = orginal[0]
    #     noised = noised[0]
 
    #     showPlots(orginal, noised)



def showPlots(orginal, noised):
    plt.subplot(121)
    plt.imshow(orginal)

    plt.subplot(122)
    plt.imshow(noised)

    plt.show()

def noisy(img):

    noise = tf.random.normal(shape=img.shape, mean=0.0, stddev=0.4, dtype=tf.dtypes.float32)

    img = img + noise

    minValue = tf.reduce_min(img)
    maxValue = tf.reduce_max(img)
    
    # Min-max feature scaling
    a = -1
    b = 1
    img = -1 + ((img - minValue)*(b - a)) / (maxValue - minValue)

    return img


def prepare_mnist_data(mnist):
    #flatten the images into vectors
    #fashion = fashion.map(lambda img, target: (tf.reshape(img, (-1,)), target))

    
    # Remove target
    mnist = mnist.map(lambda img, target: img)

    #convert data from uint8 to float32
    mnist = mnist.map(lambda img: tf.cast(img, tf.float32))

    
    #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img: (img/128.)-1.)

    # Add noise
    mnist = mnist.map(lambda img: (img, noisy(img)))
 

    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    mnist = mnist.cache()
    #shuffle, batch, prefetch
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(80)
    mnist = mnist.prefetch(20)
    #return preprocessed dataset
    return mnist

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")