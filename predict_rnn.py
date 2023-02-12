import os
import sys
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

    # Create unique save path for predictions
    k, newpath = 2, 'predictions/' + savepath
    while True:
        if not os.path.isdir(newpath):
            os.mkdir(newpath)
            break
        else:
            newpath = 'predictions/' + savepath + "_" + str(k)
            k += 1

    print(f"\n--> Created folder \"{newpath}\"")

    # Load model
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

    # Create predictions for desired stocks
    for _, stock in enumerate(stocks):
        stock = stock.replace('.', '-')
        sys.stdout.write('\rPredicting prices for: %s' % stock.ljust(4))

        predictions = torch.rand(1,0,5)   # tensor to store future predictions

        data = pd.read_csv(freq + '_prices/' + stock + '.csv', index_col=0)
        x = data[['Open', 'High', 'Low', 'Volume', 'Close']]   # input data

        mins, maxs = x.min(), x.max()   # values for normalization

        x = (x-mins)/(maxs-mins)
        x = torch.tensor(x.values)
        x = torch.unsqueeze(x, dim=0)
        
        mins = torch.tensor(mins.values)
        maxs = torch.tensor(maxs.values)

        for _ in range(steps):
            pred = model(x.float())   # model prediction for one time step
            pred = torch.unsqueeze(pred, dim=0)
            
            predictions = torch.cat((predictions, pred), dim=1)   # append predicition to full predictions tensor

            x = torch.cat((x, pred), dim=1)   # append predicition to input data for next time step

        predictions = predictions*(maxs-mins)+mins
        predictions = pd.DataFrame(predictions.squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])
        predictions.to_csv(os.path.join(newpath, stock + '.csv'), index = False)

        sys.stdout.write('\rPredicting prices for: %s - DONE' % stock.ljust(4))

    sys.stdout.flush()
    print(f"\nAll predictions saved to \"{newpath}\"")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['LSTM', 'GRU'], help='Model to be used')
    parser.add_argument('--weights', type=str, help='Path to model weights')

    parser.add_argument('--freq', type=str, choices=['daily', 'weekly', 'monthly'], default='daily', help='Predict daily, weekly, or monthly')

    parser.add_argument('--all', type=bool, default=False, help='Predict on all S&P 500 stocks')
    parser.add_argument('--stocks', nargs='+', choices=symbols, help='Stocks to predict (use tickers)')
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