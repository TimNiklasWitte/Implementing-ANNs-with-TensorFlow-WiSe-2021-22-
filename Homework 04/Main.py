import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec # row title

from MyModels import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


threshold = -1

def main():
    global threshold

    # Load dataset
    csv_file = tf.keras.utils.get_file('winequality-red.csv', 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')
    df = pd.read_csv(csv_file, delimiter=";")

    # Calc threshold: good or bad wine?
    target_name = df.columns.tolist()[-1]
    targets = df[target_name].values
    threshold = np.median( targets )

    # Split dataset
    train_ds = create_winequality_dataset( df.sample(frac=0.7) )
    test_ds = create_winequality_dataset( df.sample(frac=0.15) )
    validation_ds = create_winequality_dataset( df.sample(frac=0.15) )

    # Prepare dataset
    train_dataset = train_ds.apply(prepare_winequality_data)
    test_dataset = test_ds.apply(prepare_winequality_data)
    validation_dataset = validation_ds.apply(prepare_winequality_data)

    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.1

    # Initialize the loss:
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()

    # Optimizers:
    sgd = tf.keras.optimizers.SGD(learning_rate)
    momentum = tf.keras.optimizers.SGD(learning_rate, momentum=0.1)
    adagrad = tf.keras.optimizers.Adagrad(learning_rate)
    rms_prob = tf.keras.optimizers.RMSprop(learning_rate)
    adam = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    optimizers = [sgd, momentum, adagrad, rms_prob, adam]

    # Models:
    models = [MyModel_Regularization_L1(), MyModel_Regularization_L2(), MyModel_Dropout()]

    # Plot labels
    optimizer_name = ["Stochastic gradient descent", "Momentum", "Adagrad" , "RMSprop" , "Adam",
    "L1 Regularization", "L2 Regularization", "Dropout", "Label smoothing", "Adam + L2 Regularization + Dropout"]

    # Plot init
    rows = len(optimizer_name)
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(9, 9))
    fig.set_size_inches(18.5, 30.5)

    for row_idx in range(len(optimizer_name)):

        # Select the model, optimizer and loss

        # Use for:
        # Stochastic gradient descent (0), Momentum (1), Adagrad (2), RMSprob (3) and Adam (4)
        # -> MyModel
        if row_idx <= 4:
            model = MyModel()
            optimizer = optimizers[row_idx]

        # Label smoothing
        elif row_idx == 8:
            model = MyModel()
            optimizer = sgd
            cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)

        # Combine Adam + L2 + Dropout 
        elif row_idx == 9:
            model = MyModel_Regularization_L2_Dropout()
            optimizer = adam
            cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()

        else:
            model = models[row_idx - 5]
            optimizer = sgd

        # Initialize lists for later visualization.
        train_losses = []

        test_losses = []
        test_accuracies = []

        #testing once before we begin
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)


        # #check how model performs on train data once before we begin
        train_loss, _ = test(model, train_dataset, cross_entropy_loss)
        train_losses.append(train_loss)
        #
        # We train for num_epochs epochs.
        for epoch in range(num_epochs):
            print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

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

        # Plot result in subplot
        x = np.arange(0, len(train_losses))
        axs[row_idx, 0].plot(x, test_accuracies, 'g')
        axs[row_idx, 0].set_xlabel("Epoch")
        axs[row_idx, 0].set_ylabel("Accuracy")
        axs[row_idx, 0].set_ylim([0, 1])

        axs[row_idx, 1].plot(x, train_losses, 'r', label="Train")
        axs[row_idx, 1].plot(x, test_losses, 'b', label= "Test")
        axs[row_idx, 1].set_xlabel("Epoch")
        axs[row_idx, 1].set_ylabel("Loss")
        axs[row_idx, 1].legend(loc="upper right")


    # Plot result in subplot
    grid = plt.GridSpec(rows, cols)
    for idx, name in enumerate(optimizer_name):
        create_row_title(fig, grid[idx, ::], name)

    fig.tight_layout()
    plt.savefig("result.png", dpi=300)
    plt.show()


def prepare_winequality_data(winequality):

    # Flatten the attributes into vectors
    winequality = winequality.map(lambda attributes, target: (tf.reshape(attributes, (-1,)), target))

    # Binarize
    winequality = winequality.map(lambda attributes, target: (attributes, make_binary(target)))

    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    winequality = winequality.cache()

    # shuffle, batch, prefetch
    winequality = winequality.shuffle(1000)
    winequality = winequality.batch(15)
    winequality = winequality.prefetch(20)

    #return preprocessed dataset
    return winequality

def make_binary(target):
    global threshold
    if target > threshold:
        return 1
    else:
        return 0

def create_winequality_dataset(df):
    feature_names = df.columns.tolist()[:-1]
    target_name = df.columns.tolist()[-1]
    tf_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(df[feature_names].values, tf.float32),
                tf.cast(df[target_name].values, tf.int32)
            )
        )
    )
    return tf_dataset

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
