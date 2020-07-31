import torch.nn as nn
import torch
import math
from Encoder import TransformerEncoder
from Decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, num_layers, attention_vec_dim, num_heads, dff, dropout_rate, \
        input_vocab_size, target_vocab_size):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(num_layers, attention_vec_dim, num_heads, dff, dropout_rate, input_vocab_size)
        self.decoder = TransformerDecoder(num_layers, attention_vec_dim, num_heads, dff, dropout_rate, target_vocab_size)

        self.final_layer = nn.Linear(attention_vec_dim, target_vocab_size) # (batch_size, input_seq_len, target_vocab_size)

    def forward(self, inputs, targets, input_padding_mask, look_ahead_mask):
        enc_output = self.encoder(inputs, input_padding_mask)
        dec_output = self.decoder(targets, enc_output, look_ahead_mask, input_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output
