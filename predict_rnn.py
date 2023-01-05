import os
import torch
import argparse
import pandas as pd

from rnn import LSTM, GRU



def predict(args):
    model, weights, freq, all, stocks, steps, device, savepath = \
        args.model, args.weights, args.freq, args.all, args.stocks, args.steps, args.device, args.savepath

    if all:
        data = pd.read_csv('utils/S&P500-Info.csv', index_col=0)
        stocks = data['Symbol'].to_list()

    if not os.path.isdir('predictions'):
        os.mkdir('predictions')

    k, newpath = 2, 'predictions/' + savepath
    while True:
        if not os.path.isdir(newpath):
            os.mkdir(newpath)
            break
        else:
            newpath = savepath + "_" + str(k)
            k += 1
    os.mkdir(os.path.join(newpath))

    print(f"\n--> Created folder \"{newpath}\"")

    if model == 'LSTM':
        kwargs, state = torch.load(weights)
        model = LSTM(**kwargs)
    elif model == 'GRU':
        kwargs, state = torch.load(weights)
        model = GRU(**kwargs)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"--> Model loaded from \"{weights}\"\n")

    for _, stock in enumerate(stocks):
        print(f"Predicting future prices for \"{stock}\"")

        predictions = torch.rand(0,5)

        data = pd.read_csv(freq + '_prices/' + stock + '.csv', index_col=0)
        x = data[['Open', 'High', 'Low', 'Volume', 'Close']]    # input data

        mins, maxs = x.min(), x.max()                           # values for normalization

        x = (x-mins)/(maxs-mins)

        x = torch.tensor(x.values)
        mins = torch.tensor(mins.values)
        maxs = torch.tensor(maxs.values)

        for _ in range(steps):
            pred = model(x.float())                         # model prediction for one time step

            predictions = torch.cat((predictions, pred))    # append predicition to predictions tensor

            x = torch.cat((x, pred))                        # append predicition to input data for next time step

        predictions = predictions*(maxs-mins)+mins
        closes = pd.DataFrame(predictions.numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])
        closes.to(os.path.join(newpath, stock))

    print(f"\nAll predictions saved to \"{newpath}\"")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['LSTM', 'GRU'], help='Model to be used')
    parser.add_argument('--weights', type=str, help='Path to model weights')

    parser.add_argument('--freq', type=str, choices=['daily', 'weekly', 'monthly'], default='daily', help='Predict daily, weekly, or monthly')

    parser.add_argument('--all', type=bool, default=False, help='Predict on all S&P 500 stocks')
    parser.add_argument('--stocks', type=list, choices=symbols, default=[], help='Stocks to predict (use tickers)')
    parser.add_argument('--steps', type=int, default=25, help='Future time steps to predict')

    parser.add_argument('--device', type=str, default='cuda:0', help='device; cuda:n or cpu')

    parser.add_argument('--savepath', type=str, help='Path to save the models (best and last)')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    data = pd.read_csv('utils/S&P500-Info.csv', index_col=0)
    symbols = data['Symbol'].to_list()

    args = parse_args()
    predict(args)