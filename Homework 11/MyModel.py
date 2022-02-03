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
        self.max_input_seq_len = max_input_seq_len - 1

        self.tokenizer = tokenizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="acc"),
                        tf.keras.metrics.TopKCategoricalAccuracy(3, name="top-3-acc") 
                       ]

        self.layer_list = [
            EmbeddTokenAndPosLayer(vocabulary_size, embedding_size, max_input_seq_len),
            TransformerBlock(embedding_size),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(vocabulary_size, activation=None)
        ]

    #@tf.function
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
    
    #@tf.function
    def train_step(self, input_seq, target_token):
        
        with tf.GradientTape() as tape:
            predictions = self(input_seq, training=True)
            target_token = tf.expand_dims(target_token, -1)
            loss = self.loss_function(target_token, predictions) #+ tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics[0].update_state(loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics[1:]:
            metric.update_state(target_token,predictions)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def gen_text(self, input, k_top):
        ids = self.tokenizer.tokenize(input)
        
        num_toPredict = self.max_input_seq_len - ids.shape[0]
        if num_toPredict <= 0:
            print("Error: Text to long")
            return input

        for i in range(num_toPredict):

            # add padding
            padding_len = self.max_input_seq_len - ids.shape[0]
            padding_token_id = self.tokenizer.string_to_id("")
            padding = tf.zeros(shape=padding_len, dtype=tf.int32) + padding_token_id 

            ids_padded = tf.concat([ids, padding] , axis=-1)

            # add batch dim
            ids_padded = tf.expand_dims(ids_padded, axis=0)

            prediction_ids = self(ids_padded)

            values, indices = tf.math.top_k(prediction_ids, k=k_top, sorted=True)

            idx = tf.random.categorical(values, num_samples=1)

            idx = tf.squeeze(idx) 
            indices = tf.squeeze(indices) 
            predicted_id = indices[idx]

    
            # append predicted token
            predicted_id = tf.cast(predicted_id, tf.int32)
            predicted_id = tf.expand_dims(predicted_id, axis=-1) # add 1st dim
            ids = tf.concat([ids, predicted_id], axis=-1)
        
        return self.tokenizer.detokenize(ids)
    

    def test_step(self, dataset):

        input_seq, target_token = dataset
        target_token = tf.expand_dims(target_token, -1)

        predictions = self(input_seq, training=False)
        
        loss = self.loss_function(target_token, predictions) #+ tf.reduce_sum(self.losses)
        
        self.metrics[0].update_state(loss)
        
        for metric in self.metrics[1:]:
            metric.update_state(target_token, predictions)

        return {m.name: m.result() for m in self.metrics}