import ssl
import argparse
import pandas as pd

from urllib import request


def get_symbols(args):
    url, info, tickers, = args.url, args.info, args.tickers

    context = ssl._create_unverified_context()
    response = request.urlopen(url, context=context)
    html = response.read()

    table = pd.read_html(html)
    df = table[0]
    df.to_csv(info)
    df.to_csv(tickers, columns=['Symbol'])
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', help='wiki url of Fortune 500 companies')
    parser.add_argument('--all_info', type=str, default='S&P500-Info.csv', help='wiki url of Fortune 500 companies')
    parser.add_argument('--tickers', type=str, default='S&P500-Symbols.csv', help='wiki url of Fortune 500 companies')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    get_symbols(args)