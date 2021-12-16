import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from ConvolutionalAutoencoder.Autoencoder import * 

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

def main():

    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    train_dataset = train_ds.apply(prepare_mnist_data)
    test_dataset = test_ds.apply(prepare_mnist_data)

 
    #train_dataset = train_dataset.take(100)
    #test_dataset = test_dataset.take(1000)

    ### Hyperparameters
    num_epochs = 10
    learning_rate = 0.001

    embedding_sizes = [2,4,6,8,10]
    
    for embedding_size in embedding_sizes:

        print(f"Embedding size: {embedding_size}")


        # Initialize the model.
        for elem in train_dataset.take(1):
            _, img, _ = elem # ignore noisy and label

        model = Autoencoder(img, embedding_size)
    
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

        model.summary()
        model.encoder.summary()
        model.decoder.summary()
       
   
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

        print(f"Create plot for embedding size {embedding_size}")

        fig, axs = plt.subplots(nrows=4, ncols=50)
        total_cnt = 0
        digit_cnt = 0
        digit = 0
        done = False
        for elem in test_dataset.take(100):
            if done:
                break
        
            noised_imgs, orginal_imgs, labels = elem
            denoised_imgs = model(noised_imgs)

            embeddings = model.encoder(noised_imgs)
            embeddings = np.expand_dims(embeddings, -1)
            #embeddings = np.expand_dims(embeddings, -1)

            for i in range(16): 

                if done:
                    break 

                orginal_img = orginal_imgs[i]
                noised_img = noised_imgs[i]
                label = labels.numpy()[i]
                denoised_img = denoised_imgs[i]
                embedding = embeddings[i]
                
                height = embedding.shape[0]
                height = int(height/2)
                embedding = np.reshape(embedding, newshape=(height,2,1))


                if label == digit:
                    axs[0, total_cnt].imshow(orginal_img)
                    axs[0, total_cnt].set_title("Orginal")

                    axs[1, total_cnt].imshow(noised_img)
                    axs[1, total_cnt].set_title("Noised")

                    axs[2, total_cnt].imshow(denoised_img)
                    axs[2, total_cnt].set_title("Reconstructed")

                    axs[3, total_cnt].imshow(embedding)
                    axs[3, total_cnt].set_title("Embedding")

                    total_cnt+=1

                    digit_cnt+=1

                    if digit_cnt == 5:
                        digit_cnt = 0
                        digit += 1

                if total_cnt == 50:
                    
                    for ax in axs.flat:
                        ax.set_axis_off()

                    fig.set_size_inches(75, 6)
                    plt.tight_layout()
                
                
                    plt.savefig(f"Plots/Embedding size {embedding_size}.png")
                
                    done = True

        print("#############################")


# tsne = TSNE(n_components=2, random_state=0)

    # digits = datasets.load_digits()

    
    # X = digits.data[:500]
    # y = digits.target[:500]

    # X_2d = tsne.fit_transform(X)

    # target_ids = range(len(digits.target_names))


    # plt.figure(figsize=(6, 5))
    # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    # for i, c, label in zip(target_ids, colors, digits.target_names):
    #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    # plt.legend()
    # plt.show()

def noisy(img):

    noise = tf.random.normal(shape=img.shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)

    img = img + noise

    #minValue = tf.reduce_min(img)
    #maxValue = tf.reduce_max(img)
    
    # Min-max feature scaling
    # a = -1
    # b = 1
    # img = -1 + ((img - minValue)*(b - a)) / (maxValue - minValue)
    
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