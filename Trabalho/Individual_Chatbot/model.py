import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # layer 1 - in: input_size, out: hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)
        # layer 2 - in: hidden_size, out: hidden_size
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # layer 3 - in: hidden_size, out: num_classes
        self.l3 = nn.Linear(hidden_size, num_classes)
    
        # activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # layer 1
        out = self.l1(x)
        # activation function
        out = self.relu(out)
        
        # layer 2
        out = self.l2(out)
        # activation function
        out = self.relu(out)
        
        # layer 3
        out = self.l3(out)
        # no activation and no softmax at the end (CrossEntropyLoss)
        
        return out