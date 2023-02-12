# Combine past data and predictions
# Analyze to see what stocks will be top performers
import os
import torch
import argparse
import pandas as pd



def analyze(args):
    freq, path = args.freq, args.path

    for _, file in os.listdir(path):
        stock = file[:-4]

        predictions = pd.read_csv(os.path.join(path, file))
        data = pd.read_csv(os.path.join(freq + '_prices', stock + '.csv')).drop(['Date', 'Adj Close'])

        all_data = pd.concat([predictions, data])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--freq', type=str, choices=['daily', 'weekly', 'monthly'], default='daily', help='Predict daily, weekly, or monthly')

    parser.add_argument('--path', type=str, help='Path to predictions folder')

    parser.add_argument('--top', type=int, default=100, help='Top n% stocks to show in analysis')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    analyze(args)