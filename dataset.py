import os
import torch
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

        self.files = os.listdir(self.folder)
        new_files = []

        # set max file length ( 5 years of data )
        max = 0
        for file in self.files:
            data = pd.read_csv(os.path.join(self.folder, file), index_col=0)
            if len(data.index) > max:
                max = len(data.index)

        # remove files with less than 5 years of data
        for file in self.files:
            data = pd.read_csv(os.path.join(self.folder, file), index_col=0)
            if len(data.index) == max:
                new_files.append(file)

        self.files = new_files
        self.data = []   # stores lists of csv and partition: [AAPL.csv, n]

        # create partitions list: [AAPL.csv, n]
        for file in self.files:
            for i in range(self.splits):
                self.data.append([file, i])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        file = self.data[idx][0]        # cvs to read from
        partition = self.data[idx][1]   # partion based on number of splits

        data = pd.read_csv(os.path.join(self.folder, file), index_col=0)
        data_split = np.array_split(data, self.splits)

        # ensure all splits are of equal length
        if data_split[0].shape[0] != data_split[-1].shape[0]:
            for i, sub in enumerate(data_split):
                if sub.shape[0] != data_split[-1].shape[0]:
                    data_split[i] = data_split[i][0:-1]

        data = data_split[partition]

        x = data[['Open', 'High', 'Low', 'Volume', 'Close']]    # data

        mins, maxs = x.min(), x.max()                           # values for normalization

        x = (x-mins)/(maxs-mins)

        x = torch.tensor(x.values)
        mins = torch.tensor(mins.values)
        maxs = torch.tensor(maxs.values)

        return x, mins, maxs