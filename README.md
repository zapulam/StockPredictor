# Stock Predictor

This repo contains Python code and notebooks which can be used to train deep learning models to predict future stock prices. Currently, either an LSTM or GRU recurrent neural network can be trained and used. In the future, I hope to add functionality to train and use a time-series forcasting transformer.

## Scraping

S&P 500 data can be scraped using both the *utils/get_info.py* and *utils/get_data.py* files. By first running *get_info.py*, the names and tickers of all S&P 500 companies will be craped and saved in *S&P500-Info.csv*. Next, *get_data.py* can be run to get the past 5 years data for all S&P 500. Currently, in order to get the most recent data, the url on line 23 must be changed to reflect the most recent time period. the URL can be found at *https://finance.yahoo.com/quote/FISV/history?p=AAPL*, right clicking the download link and adjusting the JSON payload of *url* to reflect what is shown on the link.

## Training

A LSTM can be trained using the *train.py* file. Important hyperparameters are explained below...

- hidden: number of features in hidden state
- data: path to the prices data folder
- lookback: minimum range of how far to look back for each training point... for example *lookback = 60* will include 60 closing prices to train on at minimum
- savepath: path to save the models, both best and last

Training works as follows...

1. A batch of data including *n* tensors including 5 years worth of data on Low, High, open, and Close prices are loaded.
2. The data for each stock is split into sequences from the length of the desired minimum lookback all the way to the full 5 years worth of data. For instances if the length of the data is 60 rows and minimum lookback is 50, 10 sequences will be created.
3. The batch of sequences are then trained on.
4. The model is then validated using a validation set.

## Prediction

Predictions for RNNs are created using the *predict.py* file. Currently, in order to get the most recent data, the url on line 23 must be changed to reflect the most recent time period. the URL can be found at *https://finance.yahoo.com/quote/FISV/history?p=AAPL*, right clicking the download link and adjusting the JSON payload of *url* to reflect what is shown on the link. Important hyperparameters are as follows...

- weights: path to model .pth file containing model weights
- skip: if true, skips the download of most recent data
- steps: number of time steps to predict for
- savepath: path to save all .csv files

Predictions are created as follows...

1. Load all past data for a stock
2. Pass the full data through the model
3. Predict *n* steps in the future
4. Save the predictions

## Analysis

The predictions can then be analyzed using the *analyze.py* file.