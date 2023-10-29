import nltk
import numpy as np
nltk.download('punkt')

# preprocessing techniques/utils
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    '''
    sentence =  ["hello", "how", "are", "you"]
    all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag =   [ 0,     1,     0,   1,     0,     0,       0]
    # put 1 if word in sentence match with all_words, else 0
    '''
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32) # dtype=np.float32 - curso fit
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0 # 1.0 - curso fit 
    return bag
    
'''
a = "How long does shipping take?"
print(a)
a = tokenize(a)
print(a)

words = ["organize", "organizes", "organizing"]
print(words)
stemmed_words = [stem(w) for w in words]
print(stemmed_words)

sentence =  ["hello", "how", "are", "you"]
all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag =   bag_of_words(sentence, all_words)
print(bag)'''