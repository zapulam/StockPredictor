'''
Purpose: contains S&P 500 dataset class used for training
'''

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SP_500(Dataset):
    '''
    S&P 500 Dataset for training RNN to predict future close prices
    '''
    def __init__(self, folder):
        '''
        Constructor method
        '''
        self.data = []
        self.folder = folder

        all_files = os.listdir(folder)
        if '_.txt' in all_files: all_files.remove('_.txt')
        files = []

        # set max file length ( 5 years worth of data )
        max = 1257

        # remove files with less than 5 years of data
        for file in all_files:
            data = pd.read_csv(os.path.join(folder, file), index_col=0)
            if len(data.index) == max:
                files.append(file)

        # create list of files: [A.csv, AAL.csv, ...]
        for file in files:
            self.data.append(file)


    def __len__(self):
        '''
        Returns length of dataset
        
        Outputs:
            - length (int) - length of dataset
        '''
        length = len(self.data)
        return length


    def __getitem__(self, idx):
        '''
        Gets individual stock history for training 

        Inputs:
            idx (int) - index to refernece from self.data

        Outputs:
            - x (tensor) - training input data
            - mins (tensor) - minimum values for all input features
            - maxs (tensor) - maximum values for all input features
        '''

        # cvs to read from
        file = self.data[idx]
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
    