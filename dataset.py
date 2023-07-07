""" Purpose: contains S&P 500 dataset class used for training """

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SP_500(Dataset):
    def __init__(self, folder):

        self.data = []

        all_files = os.listdir(folder)
        files = []

        # set max file length ( 5 years worth of data )
        max = 1259

        # remove files with less than 5 years of data
        for file in all_files:
            data = pd.read_csv(os.path.join(folder, file), index_col=0)
            if len(data.index) == max:
                files.append(file)

        # create list of files: [A.csv, AAL.csv, ...]
        for file in files:
            self.data.append([file])


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        # cvs to read from
        file = self.data[idx][0]  
        data = pd.read_csv(os.path.join(self.folder, file), index_col=0)

        # input data
        x = data[['Open', 'High', 'Low', 'Volume', 'Close']] 

        # values for normalization
        mins, maxs = x.min(), x.max()                           

        # normalize input data
        x = (x-mins)/(maxs-mins)

        # convert to tensors
        x = torch.tensor(x.values)
        mins = torch.tensor(mins.values)
        maxs = torch.tensor(maxs.values)

        return x, mins, maxs