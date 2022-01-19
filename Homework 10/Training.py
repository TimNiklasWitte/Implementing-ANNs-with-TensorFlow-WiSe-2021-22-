import tensorflow as tf
import tensorflow_text as tf_txt

contextWindowSize = 2

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

def main():

    # s = dataGenerator()
    # s = next(s)
    # print(s)
    
    dataset = tf.data.Dataset.from_generator(dataGenerator, (tf.string, tf.string))
    dataset = dataset.take(100)

    for data in dataset:
        print(data)
    #dataset = dataset.apply(prepare_data)



    # splitter = tf_txt.RegexSplitter()
    # splitter.split(test)

    # tf.print(test)

def dataGenerator():
    global contextWindowSize

    file = open("bible.txt", "r")
    data = file.read()

    data = data.split("\n")
    data = [s for s in data if s != ""]

    data = [s.lower() for s in data]

    data = [removeSpecialCharacters(s) for s in data]
    
    data = data[10000:]

    for s in data:

        words = s.split(" ")
        if len(words) == 2*contextWindowSize + 1:
            
            input_word = words[contextWindowSize]
            
            for idx, target in enumerate(words):
                if idx != contextWindowSize:
                    #print(input_word, target) 
                    yield (input_word, target) 
            
            # print(s)
            # print("-----")

def prepare_data(data):

    #convert data from uint8 to float32
    #data = data.map(lambda img: tf.cast(img, tf.float32) )

    #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    #data = data.map(lambda img: (img/128.)-1. )

    #data = data.map(lambda img: tf.reshape(img, shape=(28,28,1)) )

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