
import os
import time
import torch
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import SP_500
from rnn import LSTM, GRU



def train(args):
    model, hidden_dim, num_layers, freq, splits, epochs, \
        lr, bs, workers, lookback, device, savepath = \
        args.model, args.hidden, args.layers, args.freq, args.splits, args.epochs, \
        args.lr, args.bs, args.workers, args.lookback, args.device, args.savepath

    if not os.path.isdir('models'):
        os.mkdir('models')

    k, newpath = 2, savepath
    while True:
        if not os.path.isdir(newpath):
            os.mkdir(newpath)
            break
        else:
            newpath = savepath + "_" + str(k)
            k += 1
    os.mkdir(os.path.join(newpath, 'weights'))

    print(f"\n--> Created folder \"{newpath}\"")

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

    print("--> Dataloaders created for training and validating")

    if model == 'LSTM':
        model = LSTM(input_dim=5, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=5)
    elif model == 'GRU':
        model = GRU(input_dim=5, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=5)
    model.to(device)

    print("--> Model created and sent to device")

    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best = 10000

    print("\nBeginning training...")
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} of {epochs}")
        start = time.time()

        t_loss = []     # training losses
        t_acc = []      # training accuracy (Close)
        v_loss = []     # validation losses
        v_acc = []      # validation accuracy (Close)

        # For each batch in the dataloader
        for _, data in enumerate(tqdm(trainloader, desc='Training', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
            inputs = data[0]

            if 'cuda' in device:
                inputs = inputs.cuda()

            seqs = []   # list of [tensor(bs, 0-n, feats), tensor(bs, n+1, feats)]

            # create sequences of min length lookback and max length inputs.shape[1]
            for i in range(lookback, inputs.shape[1]-1):
                seqs.append([inputs[:, 0: lookback, :], inputs[:, lookback+1, :]])   # [inputs[bs, 0-n, feats], inputs[bs, n+1, feats]]

            # train model for each sequence
            for _, seq in enumerate(seqs):
                pred = model(seq[0].float())

                loss = criterion(pred, seq[1].float())

                t_loss.append(loss.item())
                t_acc.extend(torch.abs(100 - (pred[:, 4] - seq[1][:, 4])).squeeze().tolist())
                #t_acc.extend(torch.abs((pred[:, 4] - seq[1][:, 4])*(data[2][:, 4]-data[1][:, 4])+data[1][:, 4]).squeeze().tolist())

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        with torch.no_grad():
            for _, data in enumerate(tqdm(valloader, desc='Validating', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):
                inputs = data[0]

                if 'cuda' in device:
                    inputs = inputs.cuda()

                for i in range(lookback, inputs.shape[1]-1):
                    if i + lookback < inputs.shape[1]:
                        seqs.append([inputs[:, 0: lookback, :], inputs[:, lookback+1, :]])

                for _, seq in enumerate(seqs):
                    pred = model(seq[0].float())

                    loss = criterion(pred, seq[1].float())

                    v_loss.append(loss.item())
                    v_acc.extend(torch.abs(100 - (pred[:, 4] - seq[1][:, 4])).squeeze().tolist())
                    #v_acc.extend(torch.abs((pred[:, 4] - seq[1][:, 4])*(data[2][:, 4]-data[1][:, 4])+data[1][:, 4]).squeeze().tolist())

        end = time.time()

        torch.save(model.state_dict(), os.path.join(savepath, "weights\last.pth"))

        avg_t_loss = sum(t_loss) / len(t_loss)
        avg_v_loss = sum(v_loss) / len(v_loss)

        avg_t_acc = sum(t_acc) / len(t_acc)
        avg_v_acc = sum(v_acc) / len(v_acc)

        if avg_v_loss < best:
            best = avg_v_loss
            torch.save(model.state_dict(), os.path.join(savepath, "weights\\best.pth"))

        print(f"Time: {round(end-start, 3)}   ", end="")
        print(f"Train Loss: {round(avg_t_loss, 5)}   ", end="")
        print(f"Train Acc: {round(avg_t_acc, 5)}   ", end="")
        print(f"Valid Loss: {round(avg_v_loss, 5)}   ", end="")
        print(f"Valid Acc: {round(avg_v_acc, 5)}   ", end="\n")
        
    print(f'\nFinished Training - Models and metrics saved to: \"{savepath}\"')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['LSTM', 'GRU'], help='Model to be used')
    parser.add_argument('--hidden', type=int, default=32, help='Number of features in hidden state')
    parser.add_argument('--layers', type=int, default=2, help='Number of recurrent layers')

    parser.add_argument('--freq', type=str, choices=['daily', 'weekly', 'monthly'], default='daily', help='Predict daily, weekly, or monthly')
    parser.add_argument('--splits', type=int, default=5, help='Number of times to split the data')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--bs', type=int, default=4, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')

    parser.add_argument('--lookback', type=int, default=60, help='Specifiy min lookback range (days, weeks, or months based on freq)')

    parser.add_argument('--device', type=str, default='cuda:0', help='device; cuda:n or cpu')

    parser.add_argument('--savepath', type=str, help='Path to save the models (best and last)')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    train(args)