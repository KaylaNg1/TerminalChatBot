import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# check if GPU is available:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
# evaluation mode
model.eval()

# creating chat bot
bot_name = "Jimmy"
print("Let's chat! type 'quit' to exit")

while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    # returns numpy array
    x = bag_of_words(sentence, all_words)
    # returns 1 arry, each with x.shape[0] elements
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()]

    # softmax outputs probabilities
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]

    # if the value of this tensor is greater than 0.75
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name}: I do not understand...')