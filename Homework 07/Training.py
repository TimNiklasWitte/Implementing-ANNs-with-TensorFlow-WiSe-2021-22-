import tensorflow as tf
import matplotlib.pyplot as plt

from MyModel import *

seq_len = 5
num_samples = 28000

RANGE_MAX = 2
RANGE_MIN = -RANGE_MAX


def main():
 
    dataset = tf.data.Dataset.from_generator(my_integration_task, (tf.float32, tf.uint8))# , output_signature=tf.TensorSpec(shape,dtype))
    
    dataset = dataset.apply(prepare_data)
 
    train_dataset = dataset.take(100000)
    test_dataset = dataset.take(5000)

    ### Hyperparameters
    num_epochs = 15
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

    fig = plt.figure()
    fig.suptitle("Accuracy and loss for training and test data")

    ax = fig.add_subplot(121)
    ax.set_xlim([0, num_epochs])
    ax.set_ylim([0, 1])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    ax.axhline(y=0.8, color='y', linestyle='--')

    ax1 = fig.add_subplot(122)
    ax1.set_xlim([0, num_epochs])
    ax1.set_ylim([0, 1])
    #ax1.legend(loc="upper right")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    x = []
    # We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

        #training (and checking in with training)
        epoch_loss_agg = []
        for input,target in train_dataset:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)

        # x.append(epoch)
     
        # ax.plot(x, test_accuracies, color='g')
        
        # ax1.plot(x, train_losses, 'r', label="Train")
        # ax1.plot(x, test_losses, 'b', label= "Test")

        # if len(x) == 1:
        #     ax1.legend(loc="upper right")    

        # fig.tight_layout()
        # fig.canvas.draw()
        # #fig.show()
        # plt.pause(0.05)
        # fig.savefig(f"./Plots/epoch {epoch}.png")



        # #track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)


    x = np.arange(0, len(train_losses))

    # First subplot
    plt.subplot(121)
    plt.plot(x, test_accuracies, 'g')

    # Dashed line for 0.8 Accuracy
    plt.axhline(y=0.8, color='y', linestyle='--')

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


def prepare_data(data):

    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    data = data.cache()

    #shuffle, batch, prefetch
    data = data.shuffle(100)
    data = data.batch(32)
    data = data.prefetch(20)
    #return preprocessed dataset
    return data


def integration_task(seq_len, num_samples):
    for _ in range(num_samples):
        data = np.random.uniform(RANGE_MIN, RANGE_MAX, seq_len)
        data = np.expand_dims(data,-1)
        sum = np.sum(data)

        if sum >= 1:
            return data, 1
        else:
            return data, 0


def my_integration_task():
    global seq_len, num_samples
    for _ in range(80000):
        data, target = integration_task(seq_len, num_samples)
        yield data, target

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")