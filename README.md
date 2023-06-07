# Stock Predictor

This repo contains Python code and notebooks which can be used to train deep learning models to predict future stock prices. Currently, either an LSTM or GRU recurrent neural network can be trained and used. In the future, I hope to add functionality to train and use a time-series forcasting transformer.

### Scraping

S&P 500 data can be scraped using both the *utils/get_info.py* and *utils/get_data.py* files. By first running *get_info.py*, the names and tickers of all S&P 500 companies will be craped and saved in *S&P500-Info.csv*. Next, *get_data.py* can be run to get the past 5 years data for all S&P 500 companies on ither a monthly, weekly, or daily basis. While all of this data is present in the repo for all frequencies, it is impotant to update the data when running future predictions. 

### Training 

Currently, either an LSTM or GRU can be trained using the *train_rnn.py* file. Important hyperparameters are explained below...

- model: LSTM or GRU
- hidden: number of features in hidden state
- layers: number of recurrent layers
- freq: which frequency of data to use, monthly, weekly, or daily
- split: number of times to split each csv file... should be kept at 1 unless there are memory issues
- lookback: minimum range of how far to look back for each training point... for example *lookback = 60* will include 60 closing prices to train on at minimum
- savepath: path to save the models, both best and last

Training works as follows...

1. A batch of data including *n* tensors including 5 years worth of data on Low, High, open, and Close prices are loaded. 
2. The data for each stock is split into sequences from the length of the desired minimum lookback all the way to the full 5 years worth of data. For instances if the length of the data is 60 rows and minimum lookback is 50, 10 sequences will be created. 
3. The batch of sequences are then trained on. 
4. The model is then validated using a validation set.

### Prediction

Predictions for RNNs are created using the *predict_rnn.py* file. Predictions can be made and saved for all S&P 500 stocks or a select few. Important hyperparameters are as follows...

- model: LSTM or GRU
- weights: path to model .pth file containing model weights
- freq: monthly, weekly, or daily prices... should match that which the model was trained on
- all: True if predictions for all stocks are desired
- steps: number of time steps to predict for
- savepath: path to save all .csv files

Predictions are created as follows...

1. Load all past data for a stock
2. Pass the full data through the model
3. Save the predictions

### Analysis

The predictions can then be analyzed using the *analyze.py* file.