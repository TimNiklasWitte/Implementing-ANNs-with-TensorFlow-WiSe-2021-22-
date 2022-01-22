from glob import glob
import tensorflow as tf
import tensorflow_text as tf_txt

import numpy as np

import tqdm
from SkipGram import *

from numpy import dot
from numpy.linalg import norm

contextWindowSize = 2
numSentences = 50000


def main():
    global vocabulary_size
    sentencesData, word_dict, _ = loadSentencesData()
    vocabulary_size, _ = countWords(sentencesData)
    
    embedding_size = 64
    NUM_EPOCHS = 10

    skipGram = SkipGram(vocabulary_size, embedding_size)

    dataset = tf.data.Dataset.from_generator(dataGenerator, (tf.int32, tf.int32))
    dataset = dataset.apply(prepare_data)

    train_size = 50000
    test_size = 1000
    train_dataset = dataset.take(train_size)

    dataset.skip(train_size)
    test_dataset = dataset.take(test_size)

    file_path = "test_logs/test"
    summary_writer = tf.summary.create_file_writer(file_path)
    with summary_writer.as_default():
        
        train_loss = skipGram.test(train_dataset)
        tf.summary.scalar(name="Train loss", data=train_loss, step=0)

        test_loss = skipGram.test(test_dataset)
        tf.summary.scalar(name="Test loss", data=test_loss, step=0)


        for epoch in range(NUM_EPOCHS):
            print(f"Epoch: {epoch}")

            epoch_loss_agg = []

            for input, target in tqdm.tqdm(train_dataset,position=0, leave=True):
       
                target = tf.expand_dims(target, -1)
                loss = skipGram.train_step(input, target)
     
                epoch_loss_agg.append(np.mean(loss, axis=0))
            
            train_loss = np.mean(epoch_loss_agg, axis=0)
            tf.summary.scalar(name="Train loss", data=train_loss, step=epoch + 1)

            test_loss = skipGram.test(test_dataset)
            tf.summary.scalar(name="Test loss", data=test_loss, step=epoch + 1)

            #########################
            # K nearest neighbord words
            #########################
            words = np.array(["holy", "father", "wine"])
            K = 10
            for word1 in words:
                
                word1_id = word_dict[word1]
                
                similar_words_cosine = [""] * K
                sims_cosine = [-1] * K

                similar_words_distance = [""] * K
                sims_distance = [99999999] * K

                for word2, word2_id in word_dict.items():

                    if word1_id != word2_id:
                        a = skipGram.embedding(word1_id).numpy()
                        b = skipGram.embedding(word2_id).numpy()

                        sim = cos_sim(a,b)
                        dist = norm(a-b)

                        if sim > np.min(sims_cosine):
                            sims_cosine[np.argmin(sims_cosine)] = sim 
                      
                            similar_words_cosine[np.argmin(sims_cosine)] = word2 

                        if dist < np.max(sims_distance):
                            sims_distance[np.argmax(sims_distance)] = dist 
                            similar_words_distance[np.argmax(sims_distance)] = word2 

                # round does not work
                sims_cosine = np.round(sims_cosine, 3)
                sims_distance = np.round(sims_distance, 3)

                # sort
                similar_words_cosine = np.array(similar_words_cosine)
                similar_words_distance = np.array(similar_words_distance)

                idxs = np.argsort(sims_cosine)[::-1] # reverse order
                sims_cosine = np.sort(sims_cosine)[::-1] # reverse order
                similar_words_cosine = similar_words_cosine[idxs]

                idxs = np.argsort(sims_distance)
                sims_distance = np.sort(sims_distance)
                similar_words_distance = similar_words_distance[idxs]


                log_header =  "| Word (1) | cos sim (1) | Word (2) | distance (2) |\n"
                log_header += "|----------|-------------|----------|----------|\n"

                log = ""
                for idx in range(len(similar_words_cosine)):
                    log += f"| {similar_words_cosine[idx]} | {sims_cosine[idx]} | {similar_words_distance[idx]} | {sims_distance[idx]} |\n"
                    
                log = log_header + log 
                tf.summary.text(name=f"{word1}", data = log, step=epoch)
         




def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

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
    numWord = 0
    for sentence in sentences:
        words = sentence.split(" ")
        for word in words:
            numWord += 1
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
    

    return sentences, word_dict, numWord

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

    return len(word_dic), word_dic

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

    s = 0.001

    sentencesData, _, numWords = loadSentencesData()

    _, word_occurence_dict = countWords(sentencesData)


    for encodedSentence in sentencesData:
    
        for idx in range(2*contextWindowSize + 1):
            input_word = encodedSentence[contextWindowSize]
            target = encodedSentence[idx]
    
            if idx != contextWindowSize:
                
                # subsampling
                z_w = word_occurence_dict[input_word] / numWords
                p_keep = ( np.sqrt(z_w/s) + 1 ) * (s/z_w)
                rand_var = np.random.uniform() # [0, 1)
              
                if p_keep < rand_var:
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