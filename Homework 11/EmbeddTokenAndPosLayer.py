import tensorflow as tf

class EmbeddTokenAndPosLayer(tf.keras.layers.Layer):
     
    def __init__(self, vocabulary_size, embedding_size, max_input_seq_len):
        super(EmbeddTokenAndPosLayer, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.max_input_seq_len = max_input_seq_len

        self.embedding_token = tf.keras.layers.Embedding(self.vocabulary_size, self.embedding_size)
        self.embedding_pos = tf.keras.layers.Embedding(self.max_input_seq_len, self.embedding_size)

    #@tf.function
    def call(self, x):
        token_embedd = self.embedding_token(x)
        pos_embedd = self.embedding_pos(tf.range(self.max_input_seq_len))

        return token_embedd + pos_embedd
    
