from numpy import dtype
import tensorflow as tf

from EmbeddTokenAndPosLayer import *
from TransformerBlock import *

class MyModel(tf.keras.Model):
    def __init__(self, tokenizer, vocabulary_size, embedding_size, max_input_seq_len):
        super(MyModel, self).__init__()
        
        self.tokenizer = tokenizer
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.max_input_seq_len = max_input_seq_len

        self.tokenizer = tokenizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="acc"),
                        tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                       ]

        self.layer_list = [
            EmbeddTokenAndPosLayer(vocabulary_size, embedding_size, max_input_seq_len),
            TransformerBlock(embedding_size),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(vocabulary_size, activation=None)
        ]

    def call(self, x, training=False):
        for layer in self.layer_list:
            try:
                x = layer(x,training)
            except:
                x = layer(x)
       
        return x

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()
    
    def train_step(self, input, target):
        pass

    def gen_text(self, input):
        ids = self.tokenizer.tokenize(input)
        

        num_toPredict = self.max_input_seq_len - ids.shape[0]
        if num_toPredict <= 0:
            print("Error: Text to long")
            return ""

        print(num_toPredict)

        for i in range(num_toPredict):

            # add padding
            padding_len = self.max_input_seq_len - ids.shape[0]
            padding = tf.zeros(shape=padding_len, dtype=tf.int32)
            ids_padded = tf.concat([ids, padding] , axis=-1)

            # add batch dim
            ids_padded = tf.expand_dims(ids_padded, axis=0)

            prediction_ids = self(ids_padded)
            predicted_id = tf.random.categorical(prediction_ids, num_samples=1)
     

            # append predicted token
            predicted_id = tf.cast(predicted_id, tf.int32)
            predicted_id = tf.squeeze(predicted_id, axis=1)

            ids = tf.concat([ids, predicted_id], axis=-1)
        
        print(self.tokenizer.detokenize(ids))
        return self.tokenizer.detokenize(ids)