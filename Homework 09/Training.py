import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

import tensorflow as tf

from Generator import *
from Discriminator import *

def main():
    dataset = tf.data.Dataset.from_generator(dataGenerator, (tf.uint8))
    dataset = dataset.apply(prepare_data)
    
    train_size = 1000
    test_size = 10
    train_dataset = dataset.take(train_size)
    dataset.skip(train_size)
    test_dataset = dataset.take(test_size)


    for epoch in range(num_epochs):



    # for elm in train_dataset.take(1):
    #     plt.imshow(elm[0])
    #     plt.show()

    #print(tf.random.normal(shape=(100,1)))
    
    #dataGenerator()

def prepare_data(data):

    #convert data from uint8 to float32
    data = data.map(lambda img: tf.cast(img, tf.float32) )

    
    #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    data = data.map(lambda img: (img/128.)-1. )

    data = data.map(lambda img: tf.reshape(img, shape=(28,28,1)) )

    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    #data = data.cache()

    #shuffle, batch, prefetch
    data = data.shuffle(200) # shuffling random generated data ;)
    data = data.batch(32)
    data = data.prefetch(20)
    #return preprocessed dataset
    return data


def dataGenerator():

    category = "candle"

    if not os.path.isfile(f"{category}.npy"):
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"  
        urllib.request.urlretrieve(url, f"{category}.npy")

    images = np.load(f"{category}.npy")

    for image in images:
        yield image

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")