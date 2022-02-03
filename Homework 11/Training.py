from ctypes import pointer
import tensorflow as tf
import tensorflow_text as tf_txt

import sentencepiece as sp

from MyModel import *

def main():
    # d_model = 4
    # pos_in_dim = np.arange(4)
    # pos_in_dim = np.expand_dims(pos_in_dim, -1)
    # pos_in_dim = np.repeat(pos_in_dim, 4, axis=1)
    
    # pos_in_seq = np.transpose(pos_in_dim)
    # print(pos_in_dim)
    # print(pos_in_seq)

    # x = tf.cast(1000, tf.float32)
    # y = tf.cast((2 * (pos_in_dim//2) / d_model), tf.float32) 
 
    # angle_rates = tf.cast(pos_in_seq, tf.float32) *  tf.math.pow(x, y) #tf.math.pow(1000, (2*pos_in_dim) / d_model)
    # print(angle_rates)
    # print(tf.sin(angle_rates[:, 0::2]))
    # print(angle_rates[:, 0::2])
    
    # angle_rates[:, 0::2] = tf.sin(angle_rates[:, 0::2])

    #

    # #= np.sin(angle_rates[:, 0::2])
    # #angle_rates[:, 1:2] #= tf.cos(angle_rates[:, 0:2])
    # exit()
    # l = EmbeddTokenAndPosLayer(10, 5, 5)

    # x = tf.constant([[7, 6, 0, 0, 1]])
    # print(l(x))

    # exit()

    path = tf.keras.utils.get_file("nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")

    sp.SentencePieceTrainer.train(input=path, model_prefix='tokenizer_model', model_type="unigram", vocab_size=2000)

    # deserialize the trained model file to load it in the correct format
    trained_tokenizer_model = tf.io.gfile.GFile('tokenizer_model.model', "rb").read()

    # load the model as a tokenizer that can be used inside a tensorflow model
    tokenizer = tf_txt.SentencepieceTokenizer(
        model=trained_tokenizer_model, out_type=tf.int32, nbest_size=-1, alpha=1, reverse=False,
        add_bos=False, add_eos=False, return_nbest=False, name=None
    )

    text = open(path, 'rb').read().decode(encoding='utf-8')
    all_ids = tokenizer.tokenize(text)

    
    all_ids = tf_txt.sliding_window(data=all_ids, width=15) #error?
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    ids_dataset = ids_dataset.apply(prepare_dataset)

    model = MyModel(tokenizer, 2000, 64, 15)
    model.gen_text("tim ist der beste")
    # for e in ids_dataset.take(1):
    #     print(e)
   

   

    # Read, then decode for py2 compat.
    #text = open(path, 'rb').read().decode(encoding='utf-8')
    
    #print(text[:250])

def prepare_dataset(dataset):

    dataset = (
        dataset
        .map(split_input_target) 
        .shuffle(10000)
        .batch(1, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    
    return dataset
 
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
