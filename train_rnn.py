
import os
import time
import torch
import argparse

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import SP_500
from rnn import LSTM, GRU



def train(args):
    model, hidden_dim, num_layers, freq, splits, epochs, lr, \
    bs, workers, max_lookback, lookback, device, savepath = \
        args.model, args.hidden, args.layers, args.freq, args.splits, args.epochs, args.lr, \
        args.batch, args.workers,args.maxlookback, args.lookback, args.device, args.savepath

    if not os.path.isdir('models'):
        os.mkdir('models')

    i, newpath = 2, savepath
    while True:
        if not os.path.isdir(newpath):
            os.mkdir(newpath)
            break
        else:
            newpath = savepath + "_" + str(i)
            i += 1
    os.mkdir(os.path.join(newpath, 'weights'))

    if freq == 'daily':
        dataset = SP_500('daily', splits)
        train, val = train_test_split(dataset, test_size=0.2, random_state=42)
    elif freq == 'weekly':
        dataset = SP_500('weekly', splits)
        train, val = train_test_split(dataset, test_size=0.2, random_state=42)
    elif freq == 'monthly':
        dataset = SP_500('monthly', splits)
        train, val = train_test_split(dataset, test_size=0.2, random_state=42)

    trainloader = DataLoader(dataset=train, batch_size=bs, shuffle=True, num_workers=workers)
    valloader = DataLoader(dataset=val, batch_size=bs, shuffle=True, num_workers=workers)

    if model == 'LSTM':
        model = LSTM(input_dim=5, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    elif model == 'GRU':
        model = GRU(input_dim=5, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
    model.to(device)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best = 10000

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} of {epochs}")
        start = time.time()

        t_loss = []   # training losses
        v_loss = []   # validation losses

        # For each batch in the dataloader
        for _, data in enumerate(tqdm(trainloader, desc='Training', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
            inputs, targets = data[0], data[1]

            if 'cuda' in device:
                inputs, targets = inputs.cuda(), targets.cuda()

            seqs = []

            if max_lookback == True:
                lookback = inputs.shape[1] - 1

            # create sequences of length lookback
            for i in range(inputs.shape[1]-lookback):
                if i + lookback < inputs.shape[1]:
                    seqs.append([inputs[:, i: i+lookback-1, :], targets[:, i+lookback, :]])   # [inputs, target] inputs shape > [batch, 0-lookback, 5]

            """
            # create sequences of length 1 to lookback
            for i in range(inputs.shape[1]-2):
                    seqs.append([inputs[:, 0: i+1, :], targets[:, i+2, :]])   # [inputs, target] inputs shape > [batch, 0-lookback, 5]
            """

            """
            # create sequences of length lookback
            for i in range(lookback, inputs.shape[1]-2):
                    seqs.append([inputs[:, i-lookback: i, :], targets[:, i+1, :]])   # [inputs, target] inputs shape > [batch, 0-lookback, 5]
            """

            # train model for each sequence
            for j, seq in enumerate(seqs):
                pred = model(seq[0].float())
                #print('\n', pred.shape, seq[1].shape, '\n')

                loss = criterion(pred, seq[1].float())
                t_loss.append(loss.item())

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        with torch.no_grad():
            for _, data in enumerate(tqdm(valloader, desc='Validating', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
                inputs, targets = data[0], data[1]

                if 'cuda' in device:
                    inputs, targets = inputs.cuda(), targets.cuda()

                pred = model(inputs[:, :-1, :].float())
                loss = criterion(pred, targets[:, -1, :])
                v_loss.append(loss.item())

        end = time.time()

        torch.save(model.state_dict(), os.path.join(savepath, "weights\last.pth"))

        firstq_t_loss = t_loss[:len(t_loss)//4]
        lastq_t_loss = t_loss[-len(t_loss)//4:]

        avg_firstq_t_loss = sum(firstq_t_loss) / len(firstq_t_loss)
        avg_lastq_t_loss = sum(lastq_t_loss) / len(lastq_t_loss)

        avg_t_loss = sum(t_loss) / len(t_loss)
        avg_v_loss = sum(v_loss) / len(v_loss)

        if avg_v_loss < best:
            best = avg_v_loss
            torch.save(model.state_dict(), os.path.join(savepath, "weights\\best.pth"))

        print(f"Time: {round(end-start, 3)}   ", end="\n")
        print(f"Train Loss: {round(avg_t_loss, 5)}   ", end="\n")
        print(f"   First Q Train Loss: {round(avg_firstq_t_loss, 5)}   ", end="\n")
        print(f"   Last Q Train Loss: {round(avg_lastq_t_loss, 5)}   ", end="\n")
        print(f"Valid Loss: {round(avg_v_loss, 5)}   ")
        
    print(f'\nFinished Training - Models and metrics saved to: {savepath}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['LSTM', 'GRU'], help='Model to be used')
    parser.add_argument('--hidden', type=int, default=32, help='Number of features in hidden state')
    parser.add_argument('--layers', type=int, default=2, help='Number of recurrent layers')

    parser.add_argument('--freq', type=str, choices=['daily', 'weekly', 'monthly'], help='Predict daily, weekly, or monthly')
    parser.add_argument('--splits', type=int, help='Number of times to split the 5 year data')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')

    parser.add_argument('--maxlookback', type=bool, default=True, help='Set lookback to maximum allowed by splits')
    parser.add_argument('--lookback', type=int, default=5, help='Specifiy lookback range (days, weeks, or months based on freq)')

    parser.add_argument('--device', type=str, default='cuda:0', help='device; cuda:n or cpu')

    parser.add_argument('--savepath', type=str, help='Path to save the models (best and last)')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    train(args)