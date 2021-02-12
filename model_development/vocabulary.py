"""
Module responsible for preparing dataset.
"""
import json
from typing import Tuple

import numpy as np

from model_development.nltk_utils import NLTKUtils


class Vocabulary:
    """
    Class responsible for preparing vocabulary from dataset (all available words and tags)
    and training dataset (features and labels).
    """
    def __init__(self):
        self.words = []
        self.tags = []
        self.tags_patterns = []
        self.ignore_words = ['.', ',', '?', '!']

    def prepare_vocabulary(self, intents_file_path: str) -> None:
        """ Creates vocabulary from the give dataset (words and tags).

                Parameters:
                    intents_file_path (str): dataset file path
        """
        unfiltered_words = []
        unfiltered_tags = []

        with open(intents_file_path, 'r') as f:
            intents = json.load(f)

            for intent in intents['intents']:
                tag = intent['tag']
                unfiltered_tags.append(tag)

                for pattern in intent['patterns']:
                    words = NLTKUtils.tokenize(pattern)
                    unfiltered_words.extend(words)
                    self.tags_patterns.append((words, tag))

        self.words = sorted(set([NLTKUtils.stem(word) for word in unfiltered_words if word not in self.ignore_words]))
        self.tags = sorted(set(unfiltered_tags))

    def prepare_data_from_vocabulary(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Creates vocabulary from the give dataset (words and tags).

                Returns:
                   Features and labels from dataset (X, y)
       """
        X_train = []
        y_train = []

        for (pattern, tag) in self.tags_patterns:
            bag = NLTKUtils.bag_of_words(pattern, self.words)
            X_train.append(bag)

            label = self.tags.index(tag)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        return X_train, y_train
