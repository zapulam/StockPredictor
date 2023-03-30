# predict for a single stock, taking recent data directly from internet
import os
import sys
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from rnn import LSTM, GRU



def daily_predict(args):
    # Load model
    if args.model == 'LSTM':
        kwargs, state = torch.load(args.weights)
        model = LSTM(**kwargs)
    elif args.model == 'GRU':
        kwargs, state = torch.load(args.weights)
        model = GRU(**kwargs)
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    for stock in args.stocks:
        data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/{x}?period1=1512000000&period2=1669766400&interval=1d&events=history&includeAdjustedClose=true'.format(x=stock))
        
        sys.stdout.write('\rPredicting prices for: %s' % stock.ljust(4))

        predictions = torch.rand(1,0,5)   # tensor to store future predictions

        input = data[['Open', 'High', 'Low', 'Volume', 'Close']]   # input data
        mins, maxs = input.min(), input.max()   # values for normalization
        input = (input-mins)/(maxs-mins)
        input = torch.tensor(input.values)
        input = torch.unsqueeze(input, dim=0)
        
        mins = torch.tensor(mins.values)
        maxs = torch.tensor(maxs.values)

        x = input

        for _ in range(args.steps):
            pred = model(x.float())   # model prediction for one time step
            pred = torch.unsqueeze(pred, dim=0)
            
            predictions = torch.cat((predictions, pred), dim=1)   # append predicition to full predictions tensor

            x = torch.cat((x, pred), dim=1)   # append predicition to input data for next time step

        predictions = predictions*(maxs-mins)+mins
        predictions = pd.DataFrame(predictions.squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])
        input = input*(maxs-mins)+mins
        input = pd.DataFrame(input.squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])

        all_data = pd.concat([input, predictions], ignore_index=True)

        plt.figure(figsize=(8, 4))
        plt.title("{x} Stock Prices".format(x=stock))
        plt.gcf().autofmt_xdate()
        plt.ylabel("Close")
        plt.xlabel("Time Step")
        plt.plot(all_data.loc[0:len(input.index)-1, "Close"], color='cornflowerblue')
        plt.plot(all_data.loc[len(input.index)-1:,"Close"], color='lightcoral')
        plt.show()

        sys.stdout.write('\rPredicting prices for: %s - DONE' % stock.ljust(4))



def weekly_predict(args):
    # Load model
    if args.model == 'LSTM':
        kwargs, state = torch.load(args.weights)
        model = LSTM(**kwargs)
    elif args.model == 'GRU':
        kwargs, state = torch.load(args.weights)
        model = GRU(**kwargs)
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    for stock in args.stocks:
        data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/{x}?period1=1512000000&period2=1669766400&interval=1w&events=history&includeAdjustedClose=true'.format(x=stock))
        
        sys.stdout.write('\rPredicting prices for: %s' % stock.ljust(4))

        predictions = torch.rand(1,0,5)   # tensor to store future predictions

        input = data[['Open', 'High', 'Low', 'Volume', 'Close']]   # input data
        mins, maxs = input.min(), input.max()   # values for normalization
        input = (input-mins)/(maxs-mins)
        input = torch.tensor(input.values)
        input = torch.unsqueeze(input, dim=0)
        
        mins = torch.tensor(mins.values)
        maxs = torch.tensor(maxs.values)

        x = input

        for _ in range(args.steps):
            pred = model(x.float())   # model prediction for one time step
            pred = torch.unsqueeze(pred, dim=0)
            
            predictions = torch.cat((predictions, pred), dim=1)   # append predicition to full predictions tensor

            x = torch.cat((x, pred), dim=1)   # append predicition to input data for next time step

        predictions = predictions*(maxs-mins)+mins
        predictions = pd.DataFrame(predictions.squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])
        input = input*(maxs-mins)+mins
        input = pd.DataFrame(input.squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])

        all_data = pd.concat([input, predictions], ignore_index=True)

        plt.figure(figsize=(8, 4))
        plt.title("{x} Stock Prices".format(x=stock))
        plt.gcf().autofmt_xdate()
        plt.ylabel("Close")
        plt.xlabel("Time Step")
        plt.plot(all_data.loc[0:len(input.index)-1, "Close"], color='cornflowerblue')
        plt.plot(all_data.loc[len(input.index)-1:,"Close"], color='lightcoral')
        plt.show()

        sys.stdout.write('\rPredicting prices for: %s - DONE' % stock.ljust(4))



def monthly_predict(args):
    # Load model
    if args.model == 'LSTM':
        kwargs, state = torch.load(args.weights)
        model = LSTM(**kwargs)
    elif args.model == 'GRU':
        kwargs, state = torch.load(args.weights)
        model = GRU(**kwargs)
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    for stock in args.stocks:
        data = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/{x}?period1=1512000000&period2=1669766400&interval=1mo&events=history&includeAdjustedClose=true'.format(x=stock))
        
        sys.stdout.write('\rPredicting prices for: %s' % stock.ljust(4))

        predictions = torch.rand(1,0,5)   # tensor to store future predictions

        input = data[['Open', 'High', 'Low', 'Volume', 'Close']]   # input data
        mins, maxs = input.min(), input.max()   # values for normalization
        input = (input-mins)/(maxs-mins)
        input = torch.tensor(input.values)
        input = torch.unsqueeze(input, dim=0)
        
        mins = torch.tensor(mins.values)
        maxs = torch.tensor(maxs.values)

        x = input

        for _ in range(args.steps):
            pred = model(x.float())   # model prediction for one time step
            pred = torch.unsqueeze(pred, dim=0)
            
            predictions = torch.cat((predictions, pred), dim=1)   # append predicition to full predictions tensor

            x = torch.cat((x, pred), dim=1)   # append predicition to input data for next time step

        predictions = predictions*(maxs-mins)+mins
        predictions = pd.DataFrame(predictions.squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])
        input = input*(maxs-mins)+mins
        input = pd.DataFrame(input.squeeze().detach().numpy(), columns=['Open', 'High', 'Low', 'Volume', 'Close'])

        all_data = pd.concat([input, predictions], ignore_index=True)

        plt.figure(figsize=(8, 4))
        plt.title("{x} Stock Prices".format(x=stock))
        plt.gcf().autofmt_xdate()
        plt.ylabel("Close")
        plt.xlabel("Time Step")
        plt.plot(all_data.loc[0:len(input.index)-1, "Close"], color='cornflowerblue')
        plt.plot(all_data.loc[len(input.index)-1:,"Close"], color='lightcoral')
        plt.show()

        sys.stdout.write('\rPredicting prices for: %s - DONE' % stock.ljust(4))



def predict(args):
    print()
    if args.freq == 'daily':
        daily_predict(args)
    elif args.freq == 'weekly':
        weekly_predict(args)
    elif args.freq == 'monthly':
        monthly_predict(args)
    print()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', type=str, choices=['daily', 'weekly', 'monthly'], default='daily', help='Predict daily, weekly, or monthly')

    parser.add_argument('--model', type=str, choices=['LSTM', 'GRU'], help='Model to be used')
    parser.add_argument('--weights', type=str, help='Path to model weights')

    parser.add_argument('--device', type=str, default='cuda:0', help='device; cuda:n or cpu')

    parser.add_argument('--steps', type=int, default=25, help='Future time steps to predict')
    parser.add_argument('--stocks', nargs='+', help='Stocks to predict', required=True)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    predict(args)