from sys import platform
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from ConvolutionalAutoencoder.Autoencoder import * 

import matplotlib.pyplot as plt
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

        #############
        # Plotting
        #############

        # Loss
        x = range(0, len(train_losses))
        plt.figure()
        plt.title(f"Train and test losses - Embedding size: {embedding_size}")
        plt.plot(x, train_losses, 'r', label="Train")
        plt.plot(x, test_losses, 'b', label= "Test")
        plt.xticks(x)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"./ConvolutionalAutoencoder/Plots/Loss/EmbeddingSize_{embedding_size}.png")

        # Plot orginal, noised and denoised image
        print(f"Create plot for embedding size {embedding_size}")

        fig, axs = plt.subplots(nrows=4, ncols=50)
        total_cnt = 0
        digit_cnt = 0
        digit = 0
        plots_done = False

        embedding_done = False

        labels_list = []
        embedding_list = []

        for elem in test_dataset.take(500):
            if plots_done and embedding_done:
                break
            
            noised_imgs, orginal_imgs, labels = elem
            denoised_imgs = model(noised_imgs)

            embeddings = model.encoder(noised_imgs) 
            embeddings = np.expand_dims(embeddings, -1)
            embeddings = np.expand_dims(embeddings, -1)

            for i in range(16): 

                if plots_done and embedding_done:
                    break 

                orginal_img = orginal_imgs[i]
                noised_img = noised_imgs[i]
                label = labels.numpy()[i]
                denoised_img = denoised_imgs[i]
                embedding = embeddings[i]
                
                embedding_list.append(embedding.flatten())
             

                height = embedding.shape[0]
                height = int(height/2)
                embedding = np.reshape(embedding, newshape=(height,2,1))

                labels_list.append(label)
                

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
                    
                    plots_done = True
                
                if len(embedding_list) == 1000:
                    embedding_done = True

        fig.set_size_inches(75, 6)
        fig.suptitle(f"Original, noised, denoised image and corresponding embedding - Embedding size: {embedding_size}")
        plt.tight_layout()

        for ax in axs.flat:
            ax.set_axis_off()

        plt.savefig(f"./ConvolutionalAutoencoder/Plots/Denoising/EmbeddingSize_{embedding_size}.png")

        # Plot latent space 
        # see: https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html

        X = np.array(embedding_list)
        y = np.array(labels_list)
        
        tsne = TSNE(n_components=2, random_state=0)

        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'lime', 'orange', 'purple'
        for i, c, label in zip(range(0,10), colors, np.arange(0,11)):
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
            plt.plot(X_2d[y == i, 0], X_2d[y == i, 1], c=c)
        
        plt.title(f"Latent Space - Embedding size: {embedding_size}")
        plt.legend()

        plt.savefig(f"./ConvolutionalAutoencoder/Plots/LatentSpace/EmbeddingSize_{embedding_size}.png")

        print("#############################")



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