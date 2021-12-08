import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec  # row title

from ResNet.ResNet import *
from DenseNet.DenseNet import *


def main():
    train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)

    train_dataset = train_ds.apply(prepare_cifar10_data)
    test_dataset = test_ds.apply(prepare_cifar10_data)

    # For showcasing we only use a subset of the training and test data (generally use all of the available data!)
    # train_dataset = train_dataset.take(100)
    # test_dataset = test_dataset.take(100)

    # ### Hyperparameters
    num_epochs = 15
    learning_rate = 0.001

    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Labels for the model
    model_names = ["ResNet", "DenseNet"]
    models = [ResNet(), DenseNet()]

    # Init plot
    rows = len(model_names)
    cols = 2
    fig, axs = plt.subplots(rows, cols)

    for row_idx, model in enumerate(models):

        # Initialize lists for later visualization.
        train_losses = []

        test_losses = []
        test_accuracies = []

        # testing once before we begin
        test_loss, test_accuracy = model.test(test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        # model.summary()

        # check how model performs on train data once before we begin
        train_loss, _ = model.test(train_dataset, cross_entropy_loss)
        train_losses.append(train_loss)

        # We train for num_epochs epochs.
        for epoch in range(num_epochs):
            print(f"Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}")

            # training (and checking in with training)
            epoch_loss_agg = []
            for input, target in train_dataset:
                train_loss = model.train_step(input, target, cross_entropy_loss, optimizer)
                epoch_loss_agg.append(train_loss)

            # track training loss
            train_losses.append(tf.reduce_mean(epoch_loss_agg))

            # testing, so we can track accuracy and test loss
            test_loss, test_accuracy = model.test(test_dataset, cross_entropy_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        # Plot result in subplot
        x = np.arange(0, len(train_losses))
        axs[row_idx, 0].plot(x, test_accuracies, 'g')
        axs[row_idx, 0].set_xlabel("Epoch")
        axs[row_idx, 0].set_ylabel("Accuracy")
        axs[row_idx, 0].set_ylim([0, 1])

        axs[row_idx, 1].plot(x, train_losses, 'r', label="Train")
        axs[row_idx, 1].plot(x, test_losses, 'b', label="Test")
        axs[row_idx, 1].set_xlabel("Epoch")
        axs[row_idx, 1].set_ylabel("Loss")
        axs[row_idx, 1].legend(loc="upper right")

    # Plot result in subplot
    grid = plt.GridSpec(rows, cols)
    for idx, name in enumerate(model_names):
        create_row_title(fig, grid[idx, ::], name)

    fig.tight_layout()
    plt.savefig("result.png", dpi=300)
    plt.show()


def prepare_cifar10_data(cifar10: tf.data.Dataset) -> tf.data.Dataset:
    # Convert data from uint8 to float32
    cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    cifar10 = cifar10.map(lambda img, target: ((img / 128.) - 1., target))

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


def create_row_title(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
