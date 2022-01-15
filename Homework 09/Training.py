from math import gamma
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

    
    dataset = tf.data.Dataset.from_generator(dataGenerator, (tf.uint8))
    dataset = dataset.apply(prepare_data)
    
    train_size = 141000 # Total size: 141545
    test_size = 350
    train_dataset = dataset.take(train_size)

    dataset.skip(train_size)
    test_dataset = dataset.take(test_size)
    
    NUM_EPOCHS = 10
    gan = GAN()
    

    # Testing once before we begin
    test_loss_g, test_loss_d, test_accuracy_real, test_accuracy_fake  = gan.test(test_dataset)
    tf.summary.scalar(name="Test loss generator", data=test_loss_g, step=0)
    tf.summary.scalar(name="Test loss discriminator", data=test_loss_d, step=0)
    tf.summary.scalar(name="Test accuracy real", data=test_accuracy_real, step=0)
    tf.summary.scalar(name="Test accuracy fake", data=test_accuracy_fake, step=0)

    # Check how model performs on train data once before we begin
    train_loss_g, train_loss_d, _, _  = gan.test(train_dataset.take(200)) # approx 
    tf.summary.scalar(name="Train loss generator", data=train_loss_g, step=0)
    tf.summary.scalar(name="Train loss discriminator", data=train_loss_d, step=0)

    noise = tf.random.normal([32,100])
    fake_images = gan.G(noise, training=False)
    tf.summary.image(name="generated_images",data = fake_images, step=0, max_outputs=32)

    # We train for num_epochs epochs.
    file_path = "test_logs/test"
    summary_writer = tf.summary.create_file_writer(file_path)
    with summary_writer.as_default():
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch: {epoch}")

            
            #training (and checking in with training)
            epoch_g_loss_agg = []
            epoch_d_loss_agg = []
            
            for data in tqdm.tqdm(train_dataset,position=0, leave=True, total=train_size/32):
                g_loss, d_loss = gan.train_step(data)

                epoch_g_loss_agg.append(g_loss)
                epoch_d_loss_agg.append(d_loss)
                
            train_loss_g = np.mean(epoch_g_loss_agg)
            train_loss_d = np.mean(epoch_d_loss_agg)
            
            tf.summary.scalar(name="Train loss generator", data=test_loss_g, step=epoch + 1)
            tf.summary.scalar(name="Train loss discriminator", data=test_loss_d, step=epoch + 1)
          
            # print("Testing...")
            # # testing, so we can track accuracy and test loss
            test_loss_g, test_loss_d, test_accuracy_real, test_accuracy_fake  = gan.test(test_dataset)
            tf.summary.scalar(name="Test loss generator", data=test_loss_g, step=epoch + 1)
            tf.summary.scalar(name="Test loss discriminator", data=test_loss_d, step=epoch + 1)
            tf.summary.scalar(name="Test accuracy real", data=test_accuracy_real, step=epoch + 1)
            tf.summary.scalar(name="Test accuracy fake", data=test_accuracy_fake, step=epoch + 1)

           

            noise = tf.random.normal([32,100])
            fake_images = gan.G(noise, training=False)
            tf.summary.image(name="generated_images",data = fake_images, step=epoch + 1, max_outputs=32)


def prepare_data(data):

    #convert data from uint8 to float32
    data = data.map(lambda img: tf.cast(img, tf.float32) )

    #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    data = data.map(lambda img: (img/128.)-1. )

    data = data.map(lambda img: tf.reshape(img, shape=(28,28,1)) )

    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    data = data.cache()

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