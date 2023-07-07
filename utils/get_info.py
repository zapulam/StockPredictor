""" Purpose: creates csv file with info on all S&P 500 stocks, including ticker symbols """

import ssl
import argparse
import pandas as pd

from urllib import request


def get_info(args):
    url, info, = args.url, args.info

    context = ssl._create_unverified_context()
    response = request.urlopen(url, context=context)
    html = response.read()

    table = pd.read_html(html)
    df = table[0]
    df.to_csv(info)
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', help='wiki url of Fortune 500 companies')
    parser.add_argument('--info', type=str, default='S&P500-Info.csv', help='wiki url of Fortune 500 companies')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    get_info(args)