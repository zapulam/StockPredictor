# 💸 Stock Predictor 💸 - Authored by Zachary Pulliam ☕

This repo contains Python 🐍 code and notebooks which can be used to train deep learning models to predict future stock prices! 📈

## Setup

It is recommended to use an anaconda environment to use the code in this repo. Anaconda can be downloaded and set up using the instructions found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html).

Once Anaconda is downloaded, an environment can be set up using the following commands...

1. To create an environment with a specific version of Python...

    ```bash
    conda create -n stockpredictor python=3.9
    ```

2. Activate the stockpredictor environment...

    ```bash
    conda activate stockpredictor
    ```

3. This repository requires PyTorch so we will next install it...

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

    ... this command is for Windows, if you are not using Windows, instructions on installing Pytorch can be found [here](https://pytorch.org/get-started/locally/).

4. The remaining dependencies can be installed from the *requirements.txt* file using the following command...

    ```bash
    conda install -r requirements.txt
    ```

    The environment is now ready to use!

## Tutorial

Using the *tutorial.ipynb* Jupyter Notebook, the existing RNN in the repo can be used to make predictions for stocks of your choice, simply add them to the list at the top of the file and run the code blocks to create predictions. By changing the value of *steps* you can adjust how many days in advance the model will predict. The variable device should be set to *cuda:n* if a GPU is available and *cpu* if not. The code block is shown below...

```bash
symbols = ['AAPL', 'AMZN']      # stocks to make predictions for
device = 'cuda:0'               # device to use; cuda or cpu
steps = 25                      # future time steps to predict for
```

The Python code will automatically pull the most recent stock data for the stocks listed and then predict *n* time steps out. The predictions are then plotted along with the historical data, with the predictions in red and the historical data in red.

## Scraping

S&P 500 data can be scraped using both the files in the *utils/* folder. By first running the following command...

```bash
python utils/get_info.py
```

... this will download the names and tickers of all S&P 500 companies from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) and will be saved in *S&P500-Info.csv*. Next, running the follwoing command...

```bash
python get_data.py
```

... will get the past 5 years data for all S&P 500 stocks. Currently, in order to get the most recent data.

## Training

An LSTM can be trained using the *train.py* file using the following command...

```bash
python train.py
```

Hyperparameters are explained below...

- hidden: number of features in hidden state
- layers: number of recurrent layers
- data: path to the prices data folder
- epochs: number of training epochs
- lr: learning rate
- bs: batch size
- workers: number of worker nodes
- lookback: minimum range of how far to look back for each training point... for example *lookback = 60* will include 60 closing prices to train on at minimum
- device: compute device
- savepath: path to save the models, both best and last

Training works as follows...

1. A batch of data including *n* tensors including 5 years worth of data on Low, High, open, and Close prices are loaded.
2. The data for each stock is split into sequences from the length of the desired minimum lookback all the way to the full 5 years worth of data. For instances if the length of the data is 60 rows and minimum lookback is 50, 10 sequences will be created.
3. The batch of sequences are then trained on.
4. The model is then validated using a validation set.

## Prediction

Predictions for RNNs are created using the *predict.py* file using the following command...

```bash
python predict.py
```

Hyperparameters are explained below...

- weights: path to model .pth file containing model weights
- skip: if true, skips the download of most recent data
- steps: number of time steps to predict for
- device: compute device
- savepath: path to save all .csv files

Prediction works as follows...

1. Load all past data for a stock
2. Pass the full data through the model
3. Predict *n* steps in the future
4. Save the predictions

## Analysis

After predictions are made using the *predict.py* script, the predictions can then be analyzed using the *analyze.py* file. This can be done using the following command...

```bash
python anaylze.py --path predictions/date
```

...where date corresponds to the date that the prediction script was originally run. The analysis will concatenate the historical data with the predicitons data and then sort the stocks by the percent change between the most recent day in the historical data and the last day in the predictions data. The results are then printed to the command line in the following form...

```bash
Predicted Top 5 stock price increases...

CZR   -   % Change:  72.802       Total Change:  38.199
GM    -   % Change:  64.054       Total Change:  24.821
COF   -   % Change:  60.248       Total Change:  68.339
EXPE  -   % Change:  59.907       Total Change:  72.571
EMN   -   % Change:  53.395       Total Change:  46.192
```

...this is an example output of the script for predicitons made on 2023-07-18 for 25 days out.

## Disclaimer

While the model in this repo have been trained directly on S&P 500 stock data, the predictions produced by this model and any other model trained via the code in this repo should not be used primarily to make financial decisions. 😄
