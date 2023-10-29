import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

with open(r'Trabalho\Individual_Chatbot\intents.json') as file:
    intents = json.load(file)
    
all_words = []
tags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # using extend instead of append because we want to
        # add each individual word(array) to our list
        
        # add to xy pair (tuple) our tokenized words and tag
        xy.append((w, tag))
        
        
ignore_words = ['?', '!', '.', ','] # we don't want these in our bag of words

# stem and lower each word and remove ponctuation 
all_words = [stem(w) for w in all_words if w not in ignore_words]

# sort(sorted()) our words and remove duplicates(set)
all_words = sorted(set(all_words))
tags = sorted(set(tags)) # remove duplicates

print(f"tags: {tags}\n")
print(f"all_words: {all_words}")

# create bag of words (training data)
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag) # numbers instead of tags (transformar tudo
    #                         em números - curso fit)
    y_train.append(label) # CrossEntropyLoss (pytorch) - curso fit
    
X_train = np.array(X_train)
y_train = np.array(y_train)