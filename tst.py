import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch import nn, Tensor
from utils.positional_encoder import PositionalEncoder



class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
        input_size: int,                        # int, number of input variables. 1 if univariate.
        dec_seq_len: int,                       # int, the length of the input sequence fed to the decoder
        batch_first: bool,                      
        out_seq_len: int=58,                    
        dim_val: int=512,                       # int, aka d_model. All sub-layers in the model produce outputs of dimension dim_val
        n_encoder_layers: int=4,                # int, number of stacked encoder layers in the encoder
        n_decoder_layers: int=4,                # int, number of stacked encoder layers in the decoder
        n_heads: int=8,                         # int, the number of attention heads (aka parallel attention layers)
        dropout_encoder: float=0.2,             # float, the dropout rate of the decoder
        dropout_decoder: float=0.2,             # float, the dropout rate of the decoder
        dropout_pos_enc: float=0.1,             # float, the dropout rate of the positional encoder
        dim_feedforward_encoder: int=2048,      # int, number of neurons in the linear layer of the encoder
        dim_feedforward_decoder: int=2048,      # int, number of neurons in the linear layer of the decoder
        num_predicted_features: int=1):         # int, the number of features you want to predict

        super().__init__() 

        self.dec_seq_len = dec_seq_len

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val)

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val)  
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=num_predicted_features)

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers, 
            norm=None)


    def forward(self, 
        src: Tensor,                        # the encoder's output sequence
        tgt: Tensor,                        # the sequence to the decoder
        src_mask: Tensor=None,              # the mask for the src sequence to prevent the model from using data points from the target sequence
        tgt_mask: Tensor=None) -> Tensor:   # the mask for the tgt sequence to prevent the model fromusing data points from the target sequence

        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features

        # Pass through all the stacked encoder layers in the encoder
        src = self.encoder(src=src) # src shape: [batch_size, enc_seq_len, dim_val]


        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt) # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask)

        # Pass through linear mapping
        decoder_output = self.linear_mapping(decoder_output) # shape [batch_size, target seq len]

        return decoder_output