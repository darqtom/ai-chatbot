"""
Module for defining neural network.
"""
from typing import List

import torch.nn as nn


class ChatNet(nn.Module):
    """ Class defining neural network. """
    def __init__(self, input_size, hidden_size, num_classes):
        super(ChatNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: List[float]):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
