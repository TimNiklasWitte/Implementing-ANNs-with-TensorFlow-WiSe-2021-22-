import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
import tqdm

from GAN import *


category = "candle"

def main():
    # Download data
    if not os.path.isfile(f"{category}.npy"):
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"  
        urllib.request.urlretrieve(url, f"{category}.npy")


    file_path = "test_logs/test"
    summary_writer = tf.summary.create_file_writer(file_path)

    dataset = tf.data.Dataset.from_generator(dataGenerator, (tf.uint8))
    dataset = dataset.apply(prepare_data)
    
    # total size: 141545
    train_size = 141000 #15000
    test_size = 350
    train_dataset = dataset.take(train_size)
    dataset.skip(train_size)
    test_dataset = dataset.take(test_size)

    #plt.imshow

    #exit()

    num_epochs = 30
    
    gan = GAN()


    print("Train: D")

    for data in tqdm.tqdm(train_dataset,position=0, leave=True):
        train_loss = gan.train_step_discriminator(data, training=True)


     # Initialize lists for later visualization.
    train_losses = []

    test_losses = []
    test_accuracies = []

    #testing once before we begin
    test_loss, test_accuracy = gan.test(test_dataset)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    #check how model performs on train data once before we begin
    train_loss, _ = gan.test(train_dataset)
    train_losses.append(train_loss)

    # print(gan.D.summary())
    # print("###########")
    # print(gan.G.summary())
    # exit()

    

   

     # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

        #training (and checking in with training)
        epoch_loss_agg = []
        for data in tqdm.tqdm(train_dataset,position=0, leave=True):
            train_loss = gan.train_step_generator(data, training=True)
            epoch_loss_agg.append(train_loss)
        
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        print(f"train loss {train_losses[-1]}")
        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = gan.test(test_dataset)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if epoch % 4 == 0:
            for data in tqdm.tqdm(train_dataset,position=0, leave=True):
                gan.train_step_discriminator(data, training=True)

        with summary_writer.as_default():
            z = tf.random.normal([32,100])
            img = gan.G(z)
            tf.summary.image(name="generated_images",data = img, step=epoch, max_outputs=32)


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
    global category

    images = np.load(f"{category}.npy")
    for image in images:
        yield image

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")