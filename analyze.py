# Combine past data and predictions
# Analyze to see what stocks will be top performers
import os
import argparse
import pandas as pd


def sort(sub_li):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][1] < sub_li[j + 1][1]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li


def analyze(args):
    freq, path, top = args.freq, args.path, args.top

    changes = []

    for file in os.listdir(path):
        stock = file[:-4]

        predictions = pd.read_csv(os.path.join(path, file))
        steps = len(predictions.index)
        data = pd.read_csv(os.path.join(freq + '_prices', stock + '.csv')).drop(columns=['Date', 'Adj Close'])

        all_data = pd.concat([data, predictions], ignore_index=True)

        last = all_data.loc[len(all_data.index) - steps - 1, 'Close']
        pred = all_data.loc[len(all_data.index) - 1, 'Close']

        change = pred - last
        p_change = ((pred - last) / last) * 100

        changes.append([stock, p_change, change])

    changes = sort(changes)
    print("\nPredicted Top", top, "stock price increases...\n")

    for i in range(top):
        print(changes[i][0].ljust(5), '-   % Change: ', str(round(changes[i][1],3)).ljust(9), '   Total Change: ', str(round(changes[i][2],3)).ljust(9))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', type=str, choices=['daily', 'weekly', 'monthly'], default='daily', help='Predict daily, weekly, or monthly')
    parser.add_argument('--path', type=str, help='Path to predictions folder')
    parser.add_argument('--top', type=int, default=25, help='Top n% stocks to show in analysis')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    analyze(args)