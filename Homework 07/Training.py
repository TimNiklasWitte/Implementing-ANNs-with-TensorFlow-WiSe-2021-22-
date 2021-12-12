import tensorflow as tf
import matplotlib.pyplot as plt


from MyModel import *

seq_len = 25
num_samples = 100000

RANGE_MAX = 2
RANGE_MIN = -RANGE_MAX


def main():
    
    # Create dataset from a generator (use wrapper, see below)
    dataset = tf.data.Dataset.from_generator(my_integration_task, (tf.float32, tf.uint8))
    dataset = dataset.apply(prepare_data)
    
    train_size = 10000
    test_size = 1000
    train_dataset = dataset.take(train_size)
    dataset.skip(train_size)
    test_dataset = dataset.take(test_size)

    ### Hyperparameters
    num_epochs = 10
    learning_rate = 0.01

    # Initialize the model.
    model = MyModel()

    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
 
    # Initialize the optimizer: 
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []

    test_losses = []
    test_accuracies = []

    #testing once before we begin
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    #check how model performs on train data once before we begin
    train_loss, _ = test(model, train_dataset, cross_entropy_loss)
    train_losses.append(train_loss)
    
    model.summary()
   
    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

        #training (and checking in with training)
        epoch_loss_agg = []
        for input,target in train_dataset:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)


    x = range(0, len(train_losses))

    # First subplot
    plt.subplot(121)
    plt.plot(x, test_accuracies, 'g')
    plt.xticks(x)

    # Dashed line for 0.8 Accuracy
    plt.axhline(y=0.8, color='y', linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # Second subplot
    plt.subplot(122)
    plt.plot(x, train_losses, 'r', label="Train")
    plt.plot(x, test_losses, 'b', label= "Test")
    plt.xticks(x)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # Format
    plt.tight_layout()

    # Save and display
    plt.savefig("result.png")
    plt.show()


def prepare_data(data):

    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    #data = data.cache()

    #shuffle, batch, prefetch
    data = data.shuffle(200) # shuffling random generated data ;)
    data = data.batch(32)
    data = data.prefetch(20)
    #return preprocessed dataset
    return data


def integration_task(seq_len):
    
    """
    Creates the sequence (data) and the corresponding target (0 or 1)
    sum(sequence) >= 1 -> target = 1
    otherwise:         -> target = 0

    Args:
        seq_len length of the sequence (number of time steps)
    
    Return:
        data, target  
    """

    data = np.random.uniform(RANGE_MIN, RANGE_MAX, seq_len)
    data = np.expand_dims(data,-1)
    sum = np.sum(data)

    if sum >= 1:
        return data, 1
    else:
        return data, 0


def my_integration_task():

    """
    Wrapper around the integration_task.
    Passing arguments in tf.data.Dataset.from_generator call is ugly ;)

    Yield:
        data, target  
    """

    global seq_len, num_samples
    for _ in range(num_samples):
        data, target = integration_task(seq_len)
        yield data, target

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")