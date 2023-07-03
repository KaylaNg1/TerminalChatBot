import torch
import torch.nn as nn

# Our Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # initializes the neural network layers ( 3 in this case )
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Every nn.Module subclass implements the
        # operations on input data in the forward method.
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out