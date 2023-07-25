'''
Purpose: downloads daily historical data for all S&P 500 stocks
'''

import os
import sys
import requests
import argparse
import pandas as pd
import datetime as dt

from datetime import date
from dateutil.relativedelta import relativedelta


def get_data(args):
    '''
    Downloads daily historical data for all S&P 500 stocks, should be ran with utils as cwd
    
    Inputs:
        args (dict) - cmd line aruments for training
            - info (str) - path to S&P500-Info.csv
    '''
    info, folder = args.info, 'daily_prices'

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'} # this is chrome, you can set whatever browser you like

    now = dt.datetime.now()

    a = dt.datetime(1970,1,1,23,59,59)
    b = dt.datetime(now.year, now.month, now.day, 23, 59, 59)
    c = b - relativedelta(years=5)

    period1 = str(int((c-a).total_seconds()))   # total seconds from today since Jan. 1, 1970 subracting 5 years
    period2 = str(int((b-a).total_seconds()))   # total seconds from today since Jan. 1, 1970

    url = 'https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true'

    dir = os.getcwd()
    dir = dir.split(os.sep)
    dir = os.path.join('C:\\', *dir[1:-1], folder)
    os.makedirs(dir, exist_ok=True)

    # Get all ticker symbols from info file
    df = pd.read_csv(info)
    symbols = df['Symbol'].tolist()

    # Download historical data for each stock
    for symbol in symbols:
        sys.stdout.write('\rGetting data for: %s' % symbol.ljust(4))
        if '.' in symbol:
            symbol = symbol.replace('.', '-')
        get = requests.get(url.format(stock=symbol, period1=period1, period2=period2), headers=headers)
        if get.status_code != 404:
            data = pd.read_csv(url.format(stock=symbol, period1=period1, period2=period2))
            data.to_csv(os.path.join(dir, symbol + '.csv'), index = False)
        sys.stdout.write('\rGetting data for: %s - DONE' % symbol.ljust(5))

    sys.stdout.write('\rAll stock historical price files saved to daily_prices')
    
    
def parse_args():
    '''
    Saves cmd line arguments for training
    
    Outputs:
        args (dict) - cmd line aruments for training
            - info (str) - path to S&P500-Info.csv
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', type=str, default='S&P500-Info.csv', help='location of S&P500-Info.csv')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    get_data(args)
    