from ctypes import pointer
import tensorflow as tf
import tensorflow_text as tf_txt
import tqdm

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

    # Logging
    file_path = "test_logs/test" 
    train_summary_writer = tf.summary.create_file_writer(file_path)

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

    train_size = 2000
    test_size = 100
    train_dataset = ids_dataset.take(train_size)
    ids_dataset.skip(train_size)
    test_dataset = ids_dataset.take(test_size)

    model = MyModel(tokenizer, 2000, 64, 15)

    log(train_summary_writer, model, train_dataset, test_dataset, 0)

    NUM_EPOCHS = 100
    for epoch in range(NUM_EPOCHS):
        
        print(f"Epoch {epoch}:")

        for input_seq, target_token in tqdm.tqdm(train_dataset,position=0, leave=True):
            metrics = model.train_step(input_seq, target_token)

        # print the metrics
        print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])

        log(train_summary_writer, model, train_dataset, test_dataset, epoch + 1)



def prepare_dataset(dataset):

    dataset = (
        dataset
        .map(split_input_target) 
        .cache()
        .shuffle(10000)
        .batch(32, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    
    return dataset
 
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[-1]
    return input_text, target_text

def log(train_summary_writer, model, train_dataset, test_dataset, epoch):

    with train_summary_writer.as_default():
        
        # Log metrics
        if epoch == 0:
            for data in train_dataset.take(1000): # approx full train dataset
                metrics = model.test_step(data)

        for metric in model.metrics:
            tf.summary.scalar(f"{metric.name}_train", metric.result(), step=epoch)

        model.reset_metrics()

        for data in test_dataset:
            metrics = model.test_step(data)
            
        for metric in model.metrics:
            tf.summary.scalar(f"{metric.name}_test", metric.result(), step=epoch)

        model.reset_metrics()

        # Log texts
        txt = model.gen_text("He is ", k_top=5)
        tf.summary.text(f"He is ", txt, step=epoch)

        txt = model.gen_text("There is ", k_top=5)
        tf.summary.text(f"There is", txt, step=epoch)

        txt = model.gen_text("Human ", k_top=5)
        tf.summary.text(f"Human ", txt, step=epoch)
        

        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
