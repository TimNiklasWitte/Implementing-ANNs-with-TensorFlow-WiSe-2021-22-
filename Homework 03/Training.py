import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from MyModel import *

def main():
    train_ds, test_ds = tfds.load('genomics_ood', split=['train', 'test'], as_supervised=True)

    train_dataset = train_ds.apply(prepare_genomics_data)
    test_dataset = test_ds.apply(prepare_genomics_data)

    #For showcasing we only use a subset of the training and test data (generally use all of the available data!)
    train_dataset = train_dataset.take(100000)
    test_dataset = test_dataset.take(1000)

    ### Hyperparameters
    num_epochs = 10
    learning_rate = 0.1

    # Initialize the model.
    model = MyModel()
    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []

    test_losses = []
    test_accuracies = []

    #testing once before we begin
    test_loss, test_accuracy = model.test(test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    #check how model performs on train data once before we begin
    train_loss, _ = model.test(train_dataset, cross_entropy_loss)
    train_losses.append(train_loss)

    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f"Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}")

        #training (and checking in with training)
        epoch_loss_agg = []
        for input,target in train_dataset:
            train_loss = model.train_step(input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)
        
        #track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        #testing, so we can track accuracy and test loss
        test_loss, test_accuracy = model.test(test_dataset, cross_entropy_loss)
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


def onehotify(dna_seq, target):
   
    vocab = {'A':'1', 'C': '2', 'G':'3', 'T':'0'}
    for key in vocab.keys():
        dna_seq = tf.strings.regex_replace(dna_seq, key, vocab[key])
    
    split = tf.strings.bytes_split(dna_seq)
    dna_seq = tf.cast(tf.strings.to_number(split), tf.uint8)

    dna_seq = tf.one_hot(dna_seq, 4)
    dna_seq = tf.reshape(dna_seq, (-1,))
    

    target = tf.one_hot(target, 10)

    return dna_seq, target

def prepare_genomics_data(genomics):

    genomics = genomics.map(onehotify)

    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    #genomics = genomics.cache()

    # shuffle, batch, prefetch
    genomics = genomics.shuffle(1000)
    genomics = genomics.batch(32)
    genomics = genomics.prefetch(20)

    #return preprocessed dataset
    return genomics

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")