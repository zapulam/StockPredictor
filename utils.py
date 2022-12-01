import os
import ssl
import pandas as pd

from urllib import request


def get_SP500():        
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    context = ssl._create_unverified_context()
    response = request.urlopen(url, context=context)
    html = response.read()

    table = pd.read_html(html)
    df = table[0]
    df.to_csv('SS&P500-Info.csv')