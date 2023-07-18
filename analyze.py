'''
Purpose: combine historical data and predictions to see what stocks will be top performers in the future
'''

import os
import argparse
import pandas as pd


def sort(sub_li):
    '''
    Sorts through all predictions
    
    Inputs:
    : sub_li (list[stock (str), p_change (float), change (float)]) - list of stocks with their corresponding predicted percentage change and total change
    
    Outpts:
    : sub_li(list[stock (str), p_change (float), change (float)]) - sroted sub_li
    '''
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][1] < sub_li[j + 1][1]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li


def analyze(args):
    '''
    Combine historical data and predictions to see what stocks will be top performers in the future
    
    Inputs:
    : args (dict) - arguments passed in via argparser
        : path (str) - path to save location
        : top (int) - number of top performers to return
    '''
    path, top = args.path, args.top

    changes = []

    for file in os.listdir(path):
        stock = file[:-4]

        predictions = pd.read_csv(os.path.join(path, file))
        steps = len(predictions.index)
        data = pd.read_csv(os.path.join('daily_prices', stock + '.csv')).drop(columns=['Date', 'Adj Close'])

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
    '''
    Saves cmd line arguments for training
    
    Outputs:
    : args (dict) - cmd line aruments for training
        : path (str) - path to save location
        : top (int) - number of top performers to return
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to predictions folder')
    parser.add_argument('--top', type=int, default=50, help='Top n stocks to show in analysis')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    analyze(args)
    