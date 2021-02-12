"""
Module responsible for predicting bot response messages.
"""
import json
import random
from typing import Tuple, List

import torch

from model_development import ChatNet
from model_development import NLTKUtils


MODEL_PATH = './static/model.pth'
INTENTS_FILE_PATH = './static/intents.json'


class ChatBot:
    """ Class defining app bot, responsible for predicting response messages. """
    def __init__(self, name: str, min_probability: float):
        self.bot_name = name
        self.min_probability = min_probability

        with open(INTENTS_FILE_PATH, 'r') as f:
            self.intents = json.load(f)

        self.model, self.words, self.tags = self.read_model_state()

    def provide_answer(self, sentence: str) -> str:
        """ Predicts response message for user sentence.

                Parameters:
                    sentence (str): user message (sentence)
                Returns:
                  Bot response message
        """
        tokenized_sentence = NLTKUtils.tokenize(sentence)
        X = NLTKUtils.bag_of_words(tokenized_sentence, self.words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        probabilities = torch.softmax(output, dim=1)

        tag = self.tags[predicted.item()]
        tag_probability = probabilities[0][predicted.item()]

        if tag_probability.item() >= self.min_probability:
            response = self.get_response_by_pred_tag(tag)
            return f'{self.bot_name}: {response}'
        else:
            return f'{self.bot_name}: Sorry, I don\'t understand'

    def get_response_by_pred_tag(self, tag: str) -> str:
        """ Draws response message from collection based on predicted tag.

                Parameters:
                    tag (str): predicted tag
                Returns:
                    Drawn message
        """
        for intent in self.intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])

    @staticmethod
    def read_model_state() -> Tuple[ChatNet, List[str], List[str]]:
        """ Reads pretrained model state.

                Returns:
                    pretrained model object, list all words and tags
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_data = torch.load(MODEL_PATH, map_location=device)

        input_size = model_data['input_size']
        hidden_size = model_data['hidden_size']
        output_size = model_data['output_size']
        words = model_data['words']
        tags = model_data['tags']

        model_state = model_data['model_state']
        model = ChatNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()

        return model, words, tags
