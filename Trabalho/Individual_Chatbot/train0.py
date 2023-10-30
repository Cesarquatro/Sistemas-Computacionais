import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

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

#print(f"tags: {tags}\n")
#print(f"all_words: {all_words}")

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

# Hyperparameters
batch_size = 8

hidden_size = 8
input_size = len(X_train[0]) # first bag of words because all bags have the same size
output_size = len(tags)

learning_rate = 0.001
num_epochs = 1000


class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train) # number of samples
        self.x_data = X_train
        self.y_data = y_train
        
    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

#print(input_size, len(all_words), len(X_train[0]))

dataset = ChatDataSet()
train_loader = DataLoader(dataset = dataset, 
                          batch_size = batch_size, 
                          shuffle = True, 
                          num_workers=2)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

print(device)
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
        
print(f'final loss, loss={loss.item():.4f}')