import tensorflow as tf
import numpy as np

class SkipGram(tf.keras.layers.Layer):
     
    def __init__(self, vocabulary_size, embedding_size):
        super(SkipGram, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.build(embedding_size) 

        self.embedding = tf.keras.layers.Embedding(self.vocabulary_size, self.embedding_size)

        learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)


    def build(self, _):
        
        self.w_score = self.add_weight(
            shape=(self.vocabulary_size, self.embedding_size), initializer="random_normal", trainable=True
        )
        self.b_score = self.add_weight(
            shape=(self.embedding_size, 1), initializer="random_normal", trainable=True
        )

    @tf.function
    def call(self, x, target):
   
        embedd = self.embedding(x)
   
        return tf.nn.nce_loss(
            weights=self.w_score,               # [vocab_size, embed_size]
            biases=tf.squeeze(self.b_score),    # [vocab_size]
            labels=target,                      # [bs, 1] <=> [bs, vocab_size]
            inputs=embedd,                      # [bs, embed_size]
            num_sampled=32,                     # negative sampling: number 
            num_classes=self.vocabulary_size
        )
         
    @tf.function
    def train_step(self, input, target):
        # loss_object and optimizer_object are instances of respective tensorflow classes
        with tf.GradientTape() as tape:
            loss = self(input, target)
         
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def test(self, test_dataset):
        
        loss_agg = []
        for input, target in test_dataset:
            target = tf.expand_dims(target, -1)
            loss = self(input, target)
            loss_agg.append(loss)
            
        return np.mean(loss_agg)