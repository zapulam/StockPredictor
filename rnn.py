'''
Purpose: contains PyTorch LSTM model class
'''

import torch
import torch.nn as nn 


class LSTM(nn.Module):
    '''
    PyTorch LSTM model class
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        '''
        Constructor method
        '''
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kwargs = {'input_dim': input_dim, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'output_dim': output_dim}
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        '''
        Feed forward sequence
        
        Inputs:
            - x (tensor) - input features ('Open', 'High', 'Low', 'Volume', 'Close')
        
        Outputs:
            - out (tensor) - prediction ('Open', 'High', 'Low', 'Volume', 'Close')
        '''
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    