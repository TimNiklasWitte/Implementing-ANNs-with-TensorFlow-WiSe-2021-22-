import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

def main():
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    train_dataset = train_ds.apply(prepare_mnist_data)
    test_dataset = test_ds.apply(prepare_mnist_data)

    for elem in train_dataset.take(1):
        
        orginal, noised = elem

        orginal = orginal[0]
        noised = noised[0]
 
        showPlots(orginal, noised)



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
    mnist = mnist.batch(32)
    mnist = mnist.prefetch(20)
    #return preprocessed dataset
    return mnist

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")