import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SP_500(Dataset):
    def __init__(self, folder):

        self.folder = folder

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

        # create list of files: [AAPL.csv]
        for file in self.files:
            self.data.append([file])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        file = self.data[idx][0]        # cvs to read from

        data = pd.read_csv(os.path.join(self.folder, file), index_col=0)

        x = data[['Open', 'High', 'Low', 'Volume', 'Close']]    # data

        mins, maxs = x.min(), x.max()                           # values for normalization

        x = (x-mins)/(maxs-mins)

        x = torch.tensor(x.values)
        mins = torch.tensor(mins.values)
        maxs = torch.tensor(maxs.values)

        return x, mins, maxs