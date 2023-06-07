import os
import sys
import torch
import requests
import argparse
import pandas as pd

from rnn import LSTM


def predict(args):
    weights, steps, device, savepath = \
        args.weights, args.steps, args.device, args.savepath
    
    os.mkdir(os.path.join('predictions', savepath))

    url = 'https://query1.finance.yahoo.com/v7/finance/download/{x}?period1=1513036800&period2=1670803200&interval=1d&events=history&includeAdjustedClose=true'

    df = pd.read_csv('utils/S&P500-Info.csv')
    symbols = df['Symbol'].tolist()

    # Update stock data to morst recent
    for symbol in symbols:
        if '.' in symbol:
            symbol = symbol.replace('.', '-')
        get = requests.get(url.format(x=symbol))
        if get.status_code != 404:
            data = pd.read_csv(url.format(x=symbol))
            data.to_csv(os.path.join('daily_prices', symbol + '.csv'), index = False)

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
    kwargs, state = torch.load(weights)
    model = LSTM(**kwargs)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"--> Model loaded from \"{weights}\"\n")

    # Create predictions for desired stocks
    for _, stock in enumerate(symbols):
        stock = stock.replace('.', '-')
        sys.stdout.write('\rPredicting prices for: %s' % stock.ljust(4))

        predictions = torch.rand(1,0,5)   # tensor to store future predictions

        data = pd.read_csv('daily_prices/' + stock + '.csv', index_col=0)
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
    parser.add_argument('--weights', type=str, default='models/rnn/weights/best.pth', help='Path to model weights')
    
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