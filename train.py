# This is our training module
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
# torch.nn is a module that contains different classes that help 
# build neural network models
import torch.nn as nn
# Dataset stores the samples and their corresponding labels
# DataLoader wraps an iterable around the Dataset to enable easy access to samples
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

# opens the intents.json file with reading mode
# using keyword 'with' with the open() function will automatically close the file
# with also handles exceptions
# f is a file object
with open('intents.json', 'r') as f:
    # returns JSON object as a dictionary
    intents = json.load(f)

all_words = []
tags = []
xy = [] # hold patterns and tags as tuple elements

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        # extend puts elements of list to another list
        all_words.extend(w)
        xy.append((w,tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) # removes duplicates by converting to set()
tags = sorted(set(tags))

# creating training lists
x_train = [] # numpy list of 0s and 1s 
y_train = [] # int list of classes

# for each tuple in list xy
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag) # numerizing labels from tags
    y_train.append(label) # CrossEntropyLoss

# converting to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

# Hyperparamters
batch_size = 8
hidden_size = 8
# number of different classes
output_size = len(tags)
# number of words in bag of words
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

# custom Dataset class
class ChatDataset(Dataset):
    def __init__(self):
        # creates 3 attributes
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    # dataset[idx]
    def  __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
# num_workers makes loading a bit faster
# returns tuple
train_loader = DataLoader(dataset = dataset, 
                          batch_size = batch_size, 
                          shuffle = True)

# check if GPU is available:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# returns a Tensor ( essentially a numpy array ) with specified device
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        word = words.to(device)
        label = labels.to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # every 100th step
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}
# pth for pytorch
FILE = "data.pth"
# saves to Pickle File
torch.save(data, FILE)

print(f'training complete. file save to {FILE}')