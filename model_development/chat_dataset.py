"""
Module for defining app bot dataset.
"""
import numpy as np
import torch.utils.data


class ChatDataset(torch.utils.data.Dataset):
    """ Class responsible for defining dataset. """
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.n_samples = len(x_train)
        self.X = x_train
        self.y = y_train

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n_samples
