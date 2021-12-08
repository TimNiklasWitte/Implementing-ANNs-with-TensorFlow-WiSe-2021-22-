import numpy as np
import tensorflow as tf

from LSTM_Cell import *

seq_len = 10
num_samples = 100

RANGE_MAX = 2
RANGE_MIN = -RANGE_MAX

def prepare_data(data):

    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    data = data.cache()

    #shuffle, batch, prefetch
    data = data.shuffle(1350)
    data = data.batch(32)
    data = data.prefetch(20)
    #return preprocessed dataset
    return data

def main():

    # dataset = tf.data.Dataset.from_generator(my_integration_task, (tf.float32, tf.uint8))# , output_signature=tf.TensorSpec(shape,dtype))
    
    # dataset = dataset.apply(prepare_data)
 
    # train_dataset = dataset.take(1350)
    # test_dataset = dataset.take(180)

    # train_dataset = dataset.take(1)
    # for elem in train_dataset:
    #     print(elem)


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