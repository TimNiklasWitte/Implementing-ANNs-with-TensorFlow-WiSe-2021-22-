from ctypes import pointer
import tensorflow as tf
import tensorflow_text as tf_txt
import tqdm

import sentencepiece as sp

from TokenPredictor import *

def main():
    
    vocab_size = 6000
    embedding_size = 64
    max_seq_len = 15

    # Logging
    file_path = "test_logs/test" 
    train_summary_writer = tf.summary.create_file_writer(file_path)

    path = tf.keras.utils.get_file("nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    sp.SentencePieceTrainer.train(input=path, model_prefix='tokenizer_model', model_type="unigram", vocab_size=vocab_size)

    # deserialize the trained model file to load it in the correct format
    trained_tokenizer_model = tf.io.gfile.GFile('tokenizer_model.model', "rb").read()

    # load the model as a tokenizer that can be used inside a tensorflow model
    tokenizer = tf_txt.SentencepieceTokenizer(
        model=trained_tokenizer_model, out_type=tf.int32, nbest_size=-1, alpha=1, reverse=False,
        add_bos=False, add_eos=False, return_nbest=False, name=None
    )

    text = open(path, 'rb').read().decode(encoding='utf-8')
    all_ids = tokenizer.tokenize(text)

    
    all_ids = tf_txt.sliding_window(data=all_ids, width=max_seq_len + 1) # "+ 1" = do not forget the target token ;)
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    ids_dataset = ids_dataset.apply(prepare_dataset)

    tokenPredictor = TokenPredictor(tokenizer, vocab_size, embedding_size, max_seq_len)

    log(train_summary_writer, tokenPredictor, ids_dataset, 0)

    NUM_EPOCHS = 600
    for epoch in range(NUM_EPOCHS):
        
        print(f"Epoch {epoch}:")

        for input_seq, target_token in tqdm.tqdm(ids_dataset,position=0, leave=True):
            metrics = tokenPredictor.train_step(input_seq, target_token)

        # print the metrics
        print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])

        log(train_summary_writer, tokenPredictor, ids_dataset, epoch + 1)

        # Save weights
        if epoch % 100 == 0:
            tokenPredictor.save_weights(f"./saved_models/trainied_weights_epoch_{epoch}", save_format="tf")



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

def log(train_summary_writer, tokenPredictor, dataset, epoch):

    with train_summary_writer.as_default():
        
        # Log metrics
        if epoch == 0:
            for data in dataset.take(1000): # approx full train dataset
                metrics = tokenPredictor.test_step(data)

        for metric in tokenPredictor.metrics:
            tf.summary.scalar(f"{metric.name}_train", metric.result(), step=epoch)

        tokenPredictor.reset_metrics()

        # Log texts
        txt = tokenPredictor.predict("He is ", k_top=5)
        tf.summary.text(f"He is ", txt, step=epoch)

        txt = tokenPredictor.predict("There is ", k_top=5)
        tf.summary.text(f"There is ", txt, step=epoch)

        txt = tokenPredictor.predict("Human ", k_top=5)
        tf.summary.text(f"Human ", txt, step=epoch)
        

        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
