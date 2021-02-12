"""
Module responsible for model training.
"""
import argparse

import torch
import torch.nn as nn
import torch.utils.data

from model_development.chat_dataset import ChatDataset
from model_development.model import ChatNet
from model_development.vocabulary import Vocabulary


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
    parser.add_argument('--dataset_path', type=str, default='../intents.json')
    parser.add_argument('--trained_model_path', type=str, default='../pretrained_models/model.pth')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--hidden-layer_size', type=int, default=8, help='size of the hidden layer')
    parser.add_argument('--lr', type=float, default=0.01, help='Adam: learning rate')
    parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
    return parser.parse_args()


opt = parse_args()


def train():
    """
    Function responsible for model training.
    """
    vocabulary = Vocabulary()
    vocabulary.prepare_vocabulary(opt.dataset_path)
    X, y = vocabulary.prepare_data_from_vocabulary()

    dataset = ChatDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=opt.n_cpu)

    input_size = len(X[0])
    output_size = len(vocabulary.tags)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChatNet(input_size, opt.hidden_layer_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.n_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            optimizer.zero_grad()

            output = model(words.to(device))
            loss = criterion(output, labels.to(device))

            loss.backward()
            optimizer.step()

        if not (epoch + 1) % 1:
            print(f'epoch: {epoch + 1}/{opt.n_epochs}, loss={loss.item():.4f}')

    print(f'Final loss={loss.item:.4f}')

    model_data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": opt.hidden_size,
        "output_size": output_size,
        "words": vocabulary.words,
        "tags": vocabulary.tags
    }

    torch.save(model_data, opt.trained_model_path)

    print('Training finished.')


if __name__ == '__main__':
    train()
