
import os
import torch
import argparse

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import SP_500
from rnn import LSTM, GRU



def train(args):
    model, hidden_dim, num_layers, freq, splits, epochs, lr, bs, workers = \
        args.model, args.hidden, args.layers, args.freq, args.splits, args.epochs, args.lr, args.batch, args.workers

    if freq == 'daily':
        dataset = SP_500('daily', splits)
    elif freq == 'weekly':
        dataset = SP_500('weekly', splits)
    else:
        dataset = SP_500('monthly', splits)
    dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=workers)

    if model == 'LSTM':
        model = LSTM(input_dim=5, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    else:
        model = GRU(input_dim=5, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} of {epochs}")

        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader, desc='Training...', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
            print(data.size())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['LSTM', 'GRU'], help='Model to be used')
    parser.add_argument('--hidden', type=int, default=32, help='Number of features in hidden state')
    parser.add_argument('--layers', type=int, default=2, help='Number of recurrent layers')

    parser.add_argument('--freq', type=str, choices=['daily', 'weekly', 'monthly'], help='Predict daily, weekly, or monthly')
    parser.add_argument('--splits', type=int, help='Number of times to split the 5 year data')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)