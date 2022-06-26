import torch.nn as nn
import torch
import math
from MultiHeadAttention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, attention_vec_dim, num_heads, dff, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.mha_self_attention = MultiHeadAttention(attention_vec_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(attention_vec_dim) # norm across columns

        self.mha_encoder_decoder_attention = MultiHeadAttention(attention_vec_dim, num_heads)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(attention_vec_dim) # norm across columns

        self.feed_fwd_network = nn.Sequential(nn.Linear(attention_vec_dim, dff), \
            nn.ReLU(), nn.Linear(dff, attention_vec_dim))
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer_norm3 = nn.LayerNorm(attention_vec_dim)

    def forward(self, x, enc_output, look_ahead_mask, input_padding_mask):
        # x: (batch size, input_seq_len, emb_dim)
        z1 = self.mha_self_attention(x, x, x, look_ahead_mask) #(input_seq_len, attention_vec_dim)
        z1 = self.dropout1(z1)
        norm_resid_z1 = self.layer_norm1(x + z1) # element-wise addition

        z2 = self.mha_encoder_decoder_attention(enc_output, enc_output, norm_resid_z1, input_padding_mask)
        z2 = self.dropout2(z2)
        norm_resid_z2 = self.layer_norm2(norm_resid_z1 + z2) # element-wise addition

        ffn_out = self.feed_fwd_network(norm_resid_z2) 
        ffn_out = self.dropout3(ffn_out)
        enc_out = self.layer_norm3(norm_resid_z2 + ffn_out) #(input_seq_len, attention_vec_dim)

        return enc_out

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, attention_vec_dim, num_heads, dff, dropout_rate, target_vocab_size):
        super(TransformerDecoder, self).__init__()
        self.embedding_layer = nn.Embedding(target_vocab_size, attention_vec_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        self.decoder_layers = [DecoderLayer(attention_vec_dim, num_heads, dff, dropout_rate) \
                for _ in range(self.num_layers)]


    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        x = self.embedding_layer(x)
        #TODO: add positional encodings
        dec_output = self.dropout(x)
        for i in range(self.num_layers):
            dec_output = self.decoder_layers[i](x, enc_output, look_ahead_mask, padding_mask)
        return dec_output
