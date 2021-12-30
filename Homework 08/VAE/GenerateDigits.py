from sys import platform
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from Autoencoder import * 
import matplotlib.pyplot as plt

def main():
    
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    train_dataset = train_ds.apply(prepare_mnist_data)
    test_dataset = test_ds.apply(prepare_mnist_data)

    # train_dataset = train_dataset.take(10000)
    # test_dataset = test_dataset.take(10)

    ### Hyperparameters
    num_epochs = 10
    learning_rate = 0.001

    embedding_size = 2
    
    model = Autoencoder(embedding_size)
    
    # Initialize the loss:
    mse = tf.keras.losses.MeanSquaredError()

    # Initialize the optimizer: 
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []

    #testing once before we begin
    test_loss  = model.test(test_dataset, mse)
    test_losses.append(test_loss)

    #check how model performs on train data once before we begin
    train_loss  = model.test(train_dataset, mse)
    train_losses.append(train_loss)

    # model.summary()
    # model.encoder.summary()
    # model.decoder.summary()
       
    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with loss {test_losses[-1]}')

        #training (and checking in with training)
        epoch_loss_agg = []
        for input,target, _ in train_dataset: # ignore label
            train_loss = model.train_step(input, target, mse, optimizer)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        # testing, so we can track accuracy and test loss
        test_loss = model.test(test_dataset, mse)
        test_losses.append(test_loss)


    NUM_PLOTS = 30
    fig, axs = plt.subplots(nrows=NUM_PLOTS, ncols=NUM_PLOTS)

    for i in range(NUM_PLOTS):
        for j in range(NUM_PLOTS):

            y = int(i - NUM_PLOTS/2)
            x = int(j - NUM_PLOTS/2)

            data = np.array([x,y])
            data = np.expand_dims(data, axis=0) # add batch dim
            result = model.decoder(data)[0]

            axs[i, j].set_title(f"[{x},{y}]")
            axs[i, j].imshow(result)
    
    fig.set_size_inches(25, 25)

    # Remove axis labels
    for ax in axs.flat:
        ax.set_axis_off()

    plt.tight_layout()

    fig.savefig("GeneratedDigits.png")
    #plt.show()



def noisy(img):

    noise = tf.random.normal(shape=img.shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)

    img = img + noise
    
    img = tf.clip_by_value(img, clip_value_min=-1, clip_value_max=1)

    return img


def prepare_mnist_data(mnist):
    
    # Remove target
    #mnist = mnist.map(lambda img, target: img)

    #convert data from uint8 to float32
    mnist = mnist.map(lambda img, label: (tf.cast(img, tf.float32), label) )

    
    #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img, label: ( (img/128.)-1., label ) )

    # Add noise
    mnist = mnist.map(lambda img, label: (noisy(img), img, label))
 

    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    mnist = mnist.cache()
    #shuffle, batch, prefetch
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(16)
    mnist = mnist.prefetch(20)
    #return preprocessed dataset
    return mnist

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")