import numpy as np
import random
import torch
import torch.nn as nn
from Transformer import Transformer

def filter_and_pad(input_data, target_data, max_input_len, max_target_len):
    # data is list of sequences each of variable length, len(data) == batch_size
    # returns a numpy array
    padded_input_data = []
    padded_target_data = []
    for i in range(len(input_data)):
        input_i = input_data[i]
        target_i = target_data[i]
        if len(input_i) <= max_input_len and len(target_i) <= max_target_len:
            padded_input_data.append(np.pad(input_i, (0, max_input_len-len(input_i)), 'constant'))
            padded_target_data.append(np.pad(target_i, (0, max_target_len-len(target_i)), 'constant'))
    
    padded_input_data = np.vstack(padded_input_data)
    padded_target_data = np.vstack(padded_target_data)
    assert (padded_input_data.shape[1] == max_input_len)
    assert (padded_target_data.shape[1] == max_target_len)
    assert (padded_input_data.shape[0] == padded_target_data.shape[0])
    return padded_input_data, padded_target_data

# masking
def create_padding_mask(data):
    '''mask_per_seq_word = np.zeros((batch_size, seq_len, seq_len))
    for i in range(batch_size):
        mask_per_seq_word[i] = np.vstack([mask_per_seq[i] for _ in range(seq_len)])
  
    assert(mask_per_seq_word.shape == (batch_size, seq_len, seq_len))
    return mask_per_seq_word'''

    # 1 in places without padding, 0 in places with padding
    batch_size, seq_len = data.shape
    mask_per_seq = (data != 0).astype(int) # (batch_size, seq_len)
    assert(mask_per_seq.shape == (batch_size, seq_len))
    return mask_per_seq[:, np.newaxis, :] # (batch_size, 1, seq_len)

def create_look_ahead_mask(data):
    # 1 in places without , 0 in places with padding
    '''mask_per_seq_per_word = np.zeros((batch_size, seq_len, seq_len))
    for i in range(batch_size):
        mask_per_seq_per_word[i] = mask_per_word
    
    assert(mask_per_seq_per_word.shape == (batch_size, seq_len, seq_len))
    return mask_per_seq_per_word'''

    batch_size, seq_len = data.shape
    mask_per_word = [np.ones(i) for i in range(1, seq_len+1)]
    mask_per_word = np.vstack([np.pad(mask_per_word[i-1], (0, seq_len - i), 'constant') for i in range(1, seq_len+1)])
    assert(mask_per_word.shape == (seq_len, seq_len)) # (seq_len, seq_len)

    return mask_per_word

def create_data(num_pts, input_vocab_size, target_vocab_size, max_input_len, max_target_len):
    input_data = []
    target_data = []

    for i in range(5, num_pts+1):
        input_data.append(np.random.randint(1, input_vocab_size+1, i))
        target_data.append(np.random.randint(1, target_vocab_size+1, i + 3))

    input_data, target_data = filter_and_pad(input_data, target_data, max_input_len, max_target_len)
    '''print("input data: {0}\n".format(input_data.shape), input_data)
    print()
    print("output data: {0}\n".format(target_data.shape), target_data)
    print()'''

    input_padding_mask = create_padding_mask(input_data)
    target_padding_mask = create_padding_mask(target_data)
    '''print("input padding mask: {0}\n".format(input_padding_mask.shape), input_padding_mask)
    print()
    print("output padding mask: {0}\n".format(target_padding_mask.shape), target_padding_mask)
    print()'''

    # (batch_size, 1, target_seq_len), (target_seq_len, target_seq_len)
    look_ahead_mask = np.minimum(target_padding_mask, create_look_ahead_mask(target_data))
    '''print("look ahead mask: {0}\n".format(look_ahead_mask.shape), look_ahead_mask)
    print()'''

    return torch.from_numpy(np.vstack(input_data)).type(torch.LongTensor), torch.from_numpy(np.vstack(target_data)).type(torch.LongTensor), \
        torch.from_numpy(input_padding_mask), torch.from_numpy(look_ahead_mask)

#TODO: add positional encoding

def loss_function(pred, real, loss_obj):
    mask = torch.logical_not(torch.eq(real, 0))
    print(real[0].squeeze().shape)
    print(pred[0].squeeze().shape)

    total_loss = torch.zeros(real.shape)
    for i in range(real.shape[0]):
        total_loss[i] = loss_obj(pred[i].squeeze(), real[i].squeeze())
    print(total_loss)

    mask = mask.type(dtype=total_loss.dtype)
    total_loss *= mask

    return torch.sum(total_loss)/torch.sum(mask)

def run_epochs(batches, model):
    "Standard Training and Logging Function"
    epochs = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_obj = nn.CrossEntropyLoss(reduction='none')

    for _ in range(epochs):
        for i in range(len(batches)):
            optimizer.zero_grad()
            batch = batches[i]
            input_data, target_data, input_padding_mask, look_ahead_mask = batch 

            outputs = model.forward(input_data, target_data, input_padding_mask, look_ahead_mask).type(torch.float)

            #one_hot_encoded = nn.functional.one_hot(target_data, out.shape[2]).type(torch.float)
            #print(one_hot_encoded.shape)
            loss = loss_function(outputs, target_data, loss_obj)
            loss.backward()
            optimizer.step()
            print(loss)
    

def main():
    input_vocab_size = 20
    target_vocab_size = 30
    max_input_len = 10
    max_target_len = 13
    num_pts = 20
    input_data, target_data, input_padding_mask, look_ahead_mask = create_data(num_pts, input_vocab_size, target_vocab_size, max_input_len, max_target_len)

    num_layers = 2
    attention_vec_dim = 512
    num_heads = 8
    dff = 2048
    dropout_rate = 0.1
    transformer = Transformer(num_layers, attention_vec_dim, num_heads, dff, dropout_rate, input_vocab_size+1, target_vocab_size+1)

    batches = [(input_data, target_data, input_padding_mask, look_ahead_mask)]
    run_epochs(batches, transformer)

main()