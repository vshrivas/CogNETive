import torch.nn as nn
import torch
import math
from MultiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, attention_vec_dim, num_heads, dff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(attention_vec_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(attention_vec_dim) # norm across columns

        self.feed_fwd_network = nn.Sequential(nn.Linear(attention_vec_dim, dff), \
            nn.ReLU(), nn.Linear(dff, attention_vec_dim))
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(attention_vec_dim)

    def forward(self, x, mask):
        # x: (batch size, input_seq_len, attention_vec_dim)
        z = self.multi_head_attention(x, x, x, mask) # (batch_size, input_seq_len, attention_vec_dim)
        z = self.dropout1(z)
        norm_resid_z = self.layer_norm1(x + z) # (batch_size, input_seq_len, attention_vec_dim)

        ffn_out = self.feed_fwd_network(norm_resid_z) # (batch_size, input_seq_len, attention_vec_dim)
        ffn_out = self.dropout2(ffn_out)
        enc_out = self.layer_norm2(norm_resid_z + ffn_out) # (batch_size, input_seq_len, attention_vec_dim)

        return enc_out


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, attention_vec_dim, num_heads, dff, dropout_rate, input_vocab_size):
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = nn.Embedding(input_vocab_size, attention_vec_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.encoder_layers = [EncoderLayer(attention_vec_dim, num_heads, dff, dropout_rate) \
                for _ in range(self.num_layers)]

    def forward(self, x, mask):
        # x (batch_size, input_seq_len)
        x = self.embedding_layer(x)
        #TODO: add positional encodings
        enc_output = self.dropout(x)
        for i in range(self.num_layers):
            enc_output = self.encoder_layers[i](x, mask)
        return enc_output

