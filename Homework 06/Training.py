import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from ResNet.ResNet import *
from DenseNet.DenseNet import *

from Util import *


def main():
    train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)

    train_dataset = train_ds.apply(prepare_cifar10_data)
    test_dataset = test_ds.apply(prepare_cifar10_data)


    #For showcasing we only use a subset of the training and test data (generally use all of the available data!)
    #train_dataset = train_dataset.take(1000)
    #test_dataset = test_dataset.take(100)

    # ### Hyperparameters
    num_epochs = 10
    learning_rate = 0.001

    # Initialize the model.
    model = DenseNet()
   
    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []

    test_losses = []
    test_accuracies = []

    #testing once before we begin
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    model.summary()

    #check how model performs on train data once before we begin
    train_loss, _ = test(model, train_dataset, cross_entropy_loss)
    train_losses.append(train_loss)

    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f"Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}")

        #training (and checking in with training)
        epoch_loss_agg = []
        for input,target in train_dataset:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)
        
        #track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        #testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # Plot results
    plt.suptitle("Accuracy and loss for training and test data")
    x = np.arange(0, len(train_losses))

    # First subplot
    plt.subplot(121)
    plt.plot(x, test_accuracies, 'g')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # Second subplot
    plt.subplot(122)
    plt.plot(x, train_losses, 'r', label="Train")
    plt.plot(x, test_losses, 'b', label= "Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # Format
    plt.tight_layout()

    # Save and display
    plt.savefig("result.png")
    plt.show()

def prepare_cifar10_data(cifar10):

    # Convert data from uint8 to float32
    cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    cifar10 = cifar10.map(lambda img, target: ((img/128.)-1., target))

    # Create one-hot targets
    cifar10 = cifar10.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    # Cache this progress in memory, as there is no need to redo it; it is deterministic after all
    cifar10 = cifar10.cache()

    # Shuffle, batch, prefetch
    cifar10 = cifar10.shuffle(1000)
    cifar10 = cifar10.batch(32)
    cifar10 = cifar10.prefetch(20)

    #  Return preprocessed dataset
    return cifar10

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")