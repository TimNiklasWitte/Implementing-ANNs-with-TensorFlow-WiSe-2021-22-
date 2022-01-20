from glob import glob
import tensorflow as tf
import tensorflow_text as tf_txt

import numpy as np

import tqdm
from SkipGram import *


contextWindowSize = 4
numSentences = 1000


def main():
    global vocabulary_size
    sentencesData = loadSentencesData()
    vocabulary_size = countWords(sentencesData)
    

    embedding_size = 64
    NUM_EPOCHS = 10

    skipGram = SkipGram(vocabulary_size, embedding_size)

    dataset = tf.data.Dataset.from_generator(dataGenerator, (tf.int32, tf.int32))
    dataset = dataset.apply(prepare_data)

    train_size = 10
    test_size = 10
    train_dataset = dataset.take(train_size)

    dataset.skip(train_size)
    test_dataset = dataset.take(test_size)

    file_path = "test_logs/test"
    summary_writer = tf.summary.create_file_writer(file_path)
    with summary_writer.as_default():
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch: {epoch}")

            epoch_loss_agg = []

            for input, target in tqdm.tqdm(train_dataset,position=0, leave=True, total=train_size/32):
       
                target = tf.expand_dims(target, -1)
                loss = skipGram.train_step(input, target)
     
                epoch_loss_agg.append(loss)
            
            train_loss = np.mean(epoch_loss_agg)
           
            tf.summary.scalar(name="Train loss", data=train_loss, step=epoch)
         

def loadSentencesData():
    global contextWindowSize
    global numSentences

    file = open("bible.txt", "r")
    data = file.read()

    sentences = data.split("\n")
    sentences = [s for s in sentences if s != ""]

    sentences = [s.lower() for s in sentences]

    sentences = [removeSpecialCharacters(s) for s in sentences]
    
    # filtering
    sentences = [s for s in sentences if len(s.split(" ")) >= 2*contextWindowSize + 1]

    sentences = sentences[:numSentences]

    # map: word->number
    word_dict = {}
    wordId = 0
    for sentence in sentences:
        words = sentence.split(" ")
        for word in words:
            try:
                word_dict[word] = word_dict[word] 
            except:
                word_dict[word] = wordId
                wordId += 1
       
   

    for idx, sentence in enumerate(sentences):
        words = sentence.split(" ")
        
        mappedWords = []
        for word in words:
            mappedWords.append(word_dict[word])

        sentences[idx] = mappedWords
    

    return sentences # = number of words

def countWords(sentences):
    global contextWindowSize

    word_dic = {}   
    for sentence in sentences:

 
        for idx in range(2*contextWindowSize + 1):
            word = sentence[idx]
            try:
                word_dic[word] += 1
            except KeyError:
                word_dic[word] = 1

    return len(word_dic)

def removeSpecialCharacters(string):
    result = ""

    for s in string:
        if s.isalpha() or s == " ":
            result += s 
    

    result = result.replace("  ", " ")

    if len(result) != 0:
        if result[0] == " ":
            result = result[1:]

        if result[-1] == " ":
            result = result[:-1]

    return result 

def dataGenerator():
    global contextWindowSize


    sentencesData = loadSentencesData()
 
    #print(vocabulary_size)
    for encodedSentence in sentencesData:
    
        for idx in range(2*contextWindowSize + 1):
            input_word = encodedSentence[contextWindowSize]
            target = encodedSentence[idx]
    
            if idx != contextWindowSize:
                yield (input_word, target) 


def prepare_data(data):
    global vocabulary_size
    
    #create one-hot inputs and targets
    #data = data.map(lambda input, target: (tf.one_hot(input, depth=vocabulary_size), tf.one_hot(target, depth=vocabulary_size)))

    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    data = data.cache()

    #shuffle, batch, prefetch
    data = data.shuffle(200) # shuffling random generated data ;)
    data = data.batch(32)
    data = data.prefetch(20)
    #return preprocessed dataset
    return data

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")