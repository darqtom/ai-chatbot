"""
Module for text processing.
"""
from typing import List

import nltk
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()


class NLTKUtils:
    """ Class containing methods for natural language processing. """
    @staticmethod
    def stem(word: str) -> str:
        """ Finds the root form of a given word (token).

               Parameters:
                   word (str): Word (token) to stem
               Returns:
                   Stem of the word
        """
        return porter_stemmer.stem(word.lower())

    @staticmethod
    def tokenize(sentence: str) -> List[str]:
        """ Splits (tokenize) sentence into list of words (tokens).

               Parameters:
                   sentence (str): Sentence to split (tokenize)
               Returns:
                   List of words (tokens)
        """
        return word_tokenize(sentence)

    @staticmethod
    def bag_of_words(tokenized_sentence: List[str], dictionary: List[str]) -> np.ndarray:
        """ Creates bag of words array from tokenized sentence.

                Parameters:
                    tokenized_sentence (List[str]): Tokenized / split sentence
                    dictionary (List[str]): List containing all words in the custom dictionary
                Returns:
                    Bag of words array (1.0 for each word from tokenized sentence known by custom dictionary,
                    0.0 otherwise
        """
        tokenized_sentence = [porter_stemmer.stem(word) for word in tokenized_sentence]
        return np.array(
            [(1 if word in tokenized_sentence else 0) for word in dictionary], dtype=np.float32)
