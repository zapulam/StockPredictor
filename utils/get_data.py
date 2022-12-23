import os
import requests
import argparse
import pandas as pd


def get_data(args):
    frequency, info, folder = args.frequency, args.info, ''

    if frequency == 'daily':
        url = 'https://query1.finance.yahoo.com/v7/finance/download/{x}?period1=1513036800&period2=1670803200&interval=1d&events=history&includeAdjustedClose=true'
        folder = 'daily_prices'
    elif frequency == 'weekly':
        url = 'https://query1.finance.yahoo.com/v7/finance/download/{x}?period1=1513036800&period2=1670803200&interval=1wk&events=history&includeAdjustedClose=true'
        folder = 'weekly_prices'
    elif frequency == 'monthly':
        url = 'https://query1.finance.yahoo.com/v7/finance/download/{x}?period1=1513036800&period2=1670803200&interval=1mo&events=history&includeAdjustedClose=true'
        folder = 'monthly_prices'

    dir = os.getcwd()
    dir = dir.split(os.sep)
    dir = os.path.join('C:\\', *dir[1:-1], folder)
    print(dir)
    os.makedirs(dir, exist_ok=True)

    df = pd.read_csv(info)
    symbols = df['Symbol'].tolist()

    for symbol in symbols:
        print('Getting data for: ' + symbol, end=' ')
        if '.' in symbol:
            symbol = symbol.replace('.', '-')
        get = requests.get(url.format(x=symbol))
        if get.status_code != 404:
            data = pd.read_csv(url.format(x=symbol))
            data.to_csv(os.path.join(dir, symbol + '.csv'))
        print('- DONE')

    print('All stock historical price files saved to ' + folder, end='\n')
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=str, default='weekly', choices=['daily', 'weekly', 'monthly'], help='frequency of price reporting')
    parser.add_argument('--info', type=str, default='S&P500-Info.csv', help='location of S&P500-Info.csv')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    get_data(args)