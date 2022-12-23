
import os
import torch
import argparse

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import SP_500
from rnn import LSTM, GRU



def train(args):
    k = 0
    model, hidden_dim, num_layers, freq, splits, epochs, lr, \
    bs, workers, max_lookback, lookback, device = \
        args.model, args.hidden, args.layers, args.freq, args.splits, args.epochs, args.lr, \
        args.batch, args.workers,args.maxlookback, args.lookback, args.device

    if freq == 'daily':
        dataset = SP_500('daily', splits)
    elif freq == 'weekly':
        dataset = SP_500('weekly', splits)
    elif freq == 'monthly':
        dataset = SP_500('monthly', splits)
    dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=workers)

    if model == 'LSTM':
        model = LSTM(input_dim=5, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    elif model == 'GRU':
        model = GRU(input_dim=5, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    model.to(device)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} of {epochs}")

        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader, desc='Training...', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
            inputs, targets = data[0], data[1]

            if 'cuda' in device:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            seqs = []

            if max_lookback == True:
                lookback = inputs.shape[1] - 1

            for i in range(inputs.shape[1]-1 - lookback): 
                seqs.append([inputs[:, i: i+lookback, :], targets[:, i+1: i+1+lookback, :]])
            
            for seq in seqs:
                pred = model(seq[0].float())
                print('/n', seq[1].shape, pred.shape, '/n')

                loss = criterion(pred, seq[1][-1])

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()


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

    parser.add_argument('--maxlookback', type=bool, default=False, help='Set lookback to maximum allowed by splits')
    parser.add_argument('--lookback', type=int, default=5, help='Lookback range (days, weeks, or months based on freq)')

    parser.add_argument('--device', type=str, default='cuda:0', help='device; cuda:n or cpu')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)