import nltk
import numpy as np
# punkt is a package with a pretrained tokenizer

# a porter stemmer is a suffix stripping algorithm
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# nltk has built in tokenization functionality
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    sentence  = ["hello", "how", "are", "you"] } tokenized list
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"] } tags
    bag     = [0,     1,     0,    1,     0,      0,       0]
    """
    # stem tokenized_sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # can write w out numpy: 
    # bag = [0] * len(all_words)
    # for word in tokenized_sentence:
    #     if word in all_words:
    #         bag[all_words.index(word)] = 1
    # return bag

    # with numpy:
    bag = np.zeros(len(all_words), dtype = np.float32)
    # idx is current iteration index, w is the value of the element
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag