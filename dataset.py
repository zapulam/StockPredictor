import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SP_500(Dataset):
    def __init__(self, frequency, splits):
        # frequency = daily, weekly, or monthly (from 'frequency'_prices folder)
        # splits = number of times a single cvs file will be split

        self.frequency = frequency
        self.folder = frequency + '_prices'
        self.splits = splits

        self.files = os.listdir(frequency + '_prices')

        self.data = []   # stores lists of csv and partition: [AAPL.csv, n]

        for file in self.files:
            for i in range(self.splits):
                self.data.append([file, i])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        file = self.data[idx][0]
        partition = self.data[idx][1]

        data = pd.read_csv(os.path.join(self.folder, file), index_col=0)
        data_split = np.array_split(data, self.splits)
        data = data_split[partition]

        x = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = data['Close']

        x = x.to_numpy()
        y = y.to_numpy()
        y = y.reshape(-1, 1)

        return x, y