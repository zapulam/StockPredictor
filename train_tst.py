import os
import json
import time
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


def train(args):


    config = TimeSeriesTransformerConfig(
        prediction_length=prediction_length,
        context_length=prediction_length*3, # context length
        lags_sequence=lags_sequence,
        num_time_features=len(time_features) + 1, # we'll add 2 time features ("month of year" and "age", see further)
        num_static_categorical_features=1, # we have a single static categorical feature, namely time series ID
        cardinality=[len(train_dataset)], # it has 366 possible values
        embedding_dimension=[2], # the model will learn an embedding of size 2 for each of the 366 possible values
        encoder_layers=4, 
        decoder_layers=4,
    )

    model = TimeSeriesTransformerForPrediction(config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=str, default='weekly', choices=['daily', 'weekly', 'monthly'], help='frequency of price reporting')
    parser.add_argument('--epochs', type=int, defualt=10)
    parser.add_argument('--prediction_length', type=int, defualt=5, help='Horizon that the decoder of the Transformer will learn to predict for')
    parser.add_argument('--context_length', type=int, defualt=10, help='Input of the encoder')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)