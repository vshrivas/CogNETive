import torch.nn as nn
import torch
import math

class AttentionHead(nn.Module):
    def __init__(self, attention_vec_dim, attention_head_dim):
        super(AttentionHead, self).__init__()
        self.attention_vec_dim = attention_vec_dim
        self.attention_head_dim = attention_head_dim

        # weight matrices
        self.wq = nn.Linear(self.attention_vec_dim, self.attention_head_dim)
        self.wk = nn.Linear(self.attention_vec_dim, self.attention_head_dim)
        self.wv = nn.Linear(self.attention_vec_dim, self.attention_head_dim)

    def scaled_dot_prod_attention(self, q, k, v, mask=None):
        query_key_match = torch.bmm(q, k.transpose(1, 2)) # (batch_size, q_seq_len, attention_head_dim) * (batch_size, attention_head_dim, k_seq_len) = (batch_size, q_seq_len, k_seq_len)
        scaled_query_key_match = torch.div(query_key_match, math.sqrt(self.attention_head_dim)) # (batch_size, q_seq_len, k_seq_len)
        
        if mask is not None: # mask should have dim (batch_size, q_seq_len, k_seq_len)
            #print(q.shape[0], q.shape[1], k.shape[1])
            assert(mask.shape[-1] == k.shape[1])
            scaled_query_key_match = scaled_query_key_match.masked_fill(mask==0, 0) # mask would be 0 where at places we need to mask

        attention_scores = nn.functional.softmax(scaled_query_key_match, dim=-1) # (batch_size, q_seq_len, k_seq_len), compute softmax along last dim
        z = torch.bmm(attention_scores, v) # (batch_size, q_seq_len, k_seq_len) * (batch_size, v_seq_len, attention_head_dim) = (batch_size, q_seq_len, attention_head_dim)
        return z # (batch_size, q_seq_len, attention_head_dim)

    def forward(self, x_q, x_k, x_v, mask):
        #TODO: do each of these inputs have the same dimensions?
        # (batch_size, input_seq_len, attention_vec_dim) * (attention_vec_dim, attention_head_dim) = (batch_size, input_seq_len, attention_head_dim)
        q = self.wq(x_q)
        k = self.wk(x_k)
        v = self.wv(x_v)
        z = self.scaled_dot_prod_attention(q, k, v, mask)
        return z # (batch_size, input_seq_len, attention_head_dim)

class MultiHeadAttention(nn.Module):
    def __init__(self, attention_vec_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention_vec_dim = attention_vec_dim 
        self.attention_head_dim = attention_vec_dim // num_heads
        self.num_heads = num_heads

        self.attention_heads = []
        for _ in range(num_heads):
            self.attention_heads.append(AttentionHead(self.attention_vec_dim, self.attention_head_dim))

        self.wo = nn.Linear(attention_vec_dim, attention_vec_dim) 

    def forward(self, x_v, x_k, x_q, mask):
        assert (x_q.shape[2] == x_k.shape[2]) and (x_k.shape[1] == x_v.shape[1])

        z = torch.empty((x_q.size(0), x_q.size(1), 0))
        for i in range(self.num_heads):
            attention_head = self.attention_heads[i]
            z = torch.cat((z, attention_head(x_q, x_k, x_v, mask)), dim=2)
        
        assert z.size() == x_q.size() # z will be (batch_size, input_seq_len, attention_vec_dim)

        mha_output = self.wo(z)
        assert mha_output.size() == x_q.size() # mha_output will be (batch_size, input_seq_len, attention_vec_dim)
        return mha_output 

        
        
