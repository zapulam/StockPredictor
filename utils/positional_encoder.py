import torch
import torch.nn as nn 
import math
from torch import nn, Tensor



class PositionalEncoder(nn.Module):
    def __init__(
        self, 
        dropout: float=0.1,         # the dropout rate
        max_seq_len: int=5000,      # the maximum length of the input sequences
        d_model: int=512,           # The dimension of the output of sub-layers in the model
        batch_first: bool=False
        ):

        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        

    def forward(self, x: Tensor) -> Tensor:     # x: Tensor, shape [batch_size, enc_seq_len, dim_val] or [enc_seq_len, batch_size, dim_val]
        x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)