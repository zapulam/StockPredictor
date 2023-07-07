""" Purpose: downloads daily historical data for all S&P 500 stocks """

import os
import sys
import requests
import argparse
import pandas as pd


def get_data(args):
    info, folder = args.info, 'daily_prices'

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'} # This is chrome, you can set whatever browser you like

    url = 'https://query1.finance.yahoo.com/v7/finance/download/{x}?period1=1528675200&period2=1686441600&interval=1d&events=history&includeAdjustedClose=true'

    dir = os.getcwd()
    dir = dir.split(os.sep)
    dir = os.path.join('C:\\', *dir[1:-1], folder)
    os.makedirs(dir, exist_ok=True)

    df = pd.read_csv(info)
    symbols = df['Symbol'].tolist()

    for symbol in symbols:
        sys.stdout.write('\rGetting data for: %s' % symbol.ljust(4))
        if '.' in symbol:
            symbol = symbol.replace('.', '-')
        get = requests.get(url.format(x=symbol), headers=headers)
        if get.status_code != 404:
            data = pd.read_csv(url.format(x=symbol))
            data.to_csv(os.path.join(dir, symbol + '.csv'), index = False)
        sys.stdout.write('\rGetting data for: %s - DONE' % symbol.ljust(5))

    sys.stdout.write('\rAll stock historical price files saved to daily_prices')
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', type=str, default='S&P500-Info.csv', help='location of S&P500-Info.csv')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    get_data(args)