from math import gamma
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec # row title

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

    # Plot
    NUM_PLOTS_PER_ROW = 5
    fig, axs = plt.subplots(nrows=NUM_EPOCHS + 1, ncols=NUM_PLOTS_PER_ROW, figsize=(14,14))


    noise = tf.random.normal([32,100])
    fake_images = gan.G(noise, training=False)
    tf.summary.image(name="generated_images",data = fake_images, step=0, max_outputs=32)

    fake_images = fake_images[:NUM_PLOTS_PER_ROW]
    for idx, img in enumerate(fake_images):
        axs[0, idx].imshow(img, cmap='gray')

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

            fake_images = fake_images[:NUM_PLOTS_PER_ROW]
            for idx, img in enumerate(fake_images):
                axs[epoch + 1, idx].imshow(img, cmap='gray')
                

    # Plot result in subplot
    grid = plt.GridSpec(NUM_EPOCHS + 1, NUM_PLOTS_PER_ROW)
    for epoch in range(NUM_EPOCHS + 1):
        create_row_title(fig, grid[epoch, ::],f"Epoch: {epoch}")

    # Remove axis labels
    for ax in axs.flat:
        ax.set_axis_off()
    
    plt.suptitle("Generated candles by applying a GAN", fontsize=22)
    plt.tight_layout()
    plt.savefig("GeneratedCandles.png", dpi=200)
  

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

def create_row_title(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='bold')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")