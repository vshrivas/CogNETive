import numpy as np
from LSTM_Params import Params, Var

class LSTMCell:
    def __init__(self, X_size):
        self.X_size = X_size
        self.H_size = 100 # Size of the hidden layer
        self.t_steps = 25 # Number of time steps (length of the sequence) used for training
        self.learning_rate = 1e-1 # Learning rate
        self.epsilon = 1e-8
        self.weight_sd = 0.1 # Standard deviation of weights for initialization
        self.Z_size = self.H_size + self.X_size # Size of concatenate(H, X) vector
        self.params = Params(self.H_size, self.X_size, self.Z_size, self.weight_sd)

    # receives hidden state and cell state from previous timestep
    # and current state (x_t) so it can predict (x_(t + 1))
    def forwardprop(self, h_prev, c_prev, x):
        # z = concatenate h_prev and x with h_prev on top of x
        z = np.row_stack((h_prev, x))

        # forget gate layer: sigmoid outputs a number between 0 - 1,
        # for each cell state in c_prev (1 represents completely keep, 0 represents
        # completely get rid of)
        f_t = self.sigmoid(np.dot(self.params.W_f.val, z) + self.params.b_f.val)

        # input gate layer: decide what new information to store in cell state
        i_t = self.sigmoid(np.dot(self.params.W_i.val, z) + self.params.b_i.val)
        c_bar = self.tanh(np.dot(self.params.W_cbar.val, z) + self.params.b_cbar.val)
        c_t = f_t * c_prev + i_t * c_bar

        # output gate layer:
        o_t = self.sigmoid(np.dot(self.params.W_o.val, z) + self.params.b_o.val)
        h_t = o_t * self.tanh(c_t)

        # fully connected layer, before softmax (not sure if/why necessary)
        v = np.dot(self.params.W_v.val, h_t) + self.params.b_v.val
        # softmax of v to turn into prob distribution
        yhat = np.exp(v) / np.sum(np.exp(v))

        # intermediate values will need to be used in backprop
        fprop_results = {'z':z, 'f_t':f_t, 'i_t':i_t, 'c_bar':c_bar, 'c_t':c_t,
            'o_t':o_t, 'h_t':h_t, 'v':v, 'y_hat':yhat}
        return fprop_results

    def backprop(self, y, dh_next, dc_next, c_prev, fprop_results):
        dv = np.copy(fprop_results['y_hat']) # differentiate loss function w.r.t y
        dv[y] -= 1

        # softmax layer
        dWv = np.dot(dv, fprop_results['h_t'].T) # dL / dWv = dL/dv * dv/dWv = dy * h
        dbv = dv # dL / dby = dL/dy * dy/dby = dy * 1
        self.params.W_v.der += dWv
        self.params.b_v.der += dbv

        # current hidden state and cell state
        dh = np.dot(self.params.W_v.val.T, dv) + dh_next # dL/dh = dL/dv * dv/dh = dv * Wv
        dc = np.copy(dc_next)
        dc += dh * fprop_results['o_t'] * self.dtanh_arctanh(self.tanh(fprop_results['c_t']))

        # gates
        df = dc * c_prev
        df = self.dsigmoid_logit(fprop_results['f_t']) * df
        self.params.W_f.der += np.dot(df, (fprop_results['z']).T)
        self.params.b_f.der += df
        dXf = np.dot(self.params.W_f.val.T, df)

        di = dc * fprop_results['c_bar']
        di = self.dsigmoid_logit(fprop_results['i_t']) * di
        self.params.W_i.der += np.dot(di, (fprop_results['z']).T)
        self.params.b_i.der += di
        dXi = np.dot(self.params.W_i.val.T, di)

        do = dh * self.tanh(fprop_results['c_t']) # dl / do = (dl / dh)(dh / do)
        do = self.dsigmoid_logit(fprop_results['o_t']) * do
        self.params.W_o.der += np.dot(do, (fprop_results['z']).T)
        self.params.b_o.der += do
        dXo = np.dot(self.params.W_o.val.T, do)

        dc_bar = dc * fprop_results['i_t']
        dc_bar = self.dtanh_arctanh(fprop_results['c_bar']) * dc_bar
        self.params.W_cbar.der += np.dot(dc_bar, (fprop_results['z']).T)
        self.params.b_cbar.der += dc_bar
        dXc_bar = np.dot(self.params.W_cbar.val.T, dc_bar)

        dX = dXf + dXi + dXo + dXc_bar
        dh_prev = dX[:self.H_size, :]
        dc_prev = fprop_results['f_t'] * dc

        return dh_prev, dc_prev

    # inputs should be a dict from index to char
    def training_step(self, inputs, outputs, h_prev_init, c_prev_init):
        loss = 0
        fresults = dict()
        # do forward prop on all the training examples
        # calculate loss by adding to it at each step
        for tstep in range(0, len(inputs)):
            x = np.zeros((self.X_size, 1))
            x[inputs[tstep]] = 1
            if tstep == 0:
                fresults[tstep] = self.forwardprop(h_prev_init,
                c_prev_init, x)
            else:
                fresults[tstep] = self.forwardprop(fresults[tstep - 1]['h_t'],
                    fresults[tstep - 1]['c_t'], x)

            loss += -np.log(fresults[tstep]['y_hat'][outputs[tstep], 0])

        # clear gradients
        for var in self.params.get_all():
            var.der.fill(0)

        # initially these gradients are 0
        dh_next = np.zeros_like(fresults[0]['h_t']) #dh from the next character
        dc_next = np.zeros_like(fresults[0]['c_t']) #dh from the next character
        # do backward pass on all training examples
        # accumulate gradient at each
        for tstep in reversed(range(0, len(inputs))):
            if tstep == 0:
                c_prev = c_prev_init
            else:
                c_prev = fresults[tstep - 1]['c_t']
            dh_next, dc_next = self.backprop(outputs[tstep], dh_next, dc_next, c_prev,
                fresults[tstep])

        # clip gradients
        for p in self.params.get_all():
            np.clip(p.der, -1, 1, out=p.der)

        return loss, fresults[len(inputs) - 1]['h_t'], fresults[len(inputs) - 1]['c_t']

    def train(self, data, chars):
        smooth_loss = -np.log(1.0 / self.X_size) * self.t_steps
        index = 0 # location in training data
        iteration = 0
        char_to_idx = {ch:i for i,ch in enumerate(chars)}
        idx_to_char = {i:ch for i,ch in enumerate(chars)}

        while True:
            if index + self.t_steps >= len(data) or iteration == 0:
                index = 0
                h_prev = np.zeros((self.H_size, 1))
                c_prev = np.zeros((self.H_size, 1))

            inputs = ([char_to_idx[ch]
                for ch in data[index: index + self.t_steps]])
            outputs = ([char_to_idx[ch]
                for ch in data[index + 1: index + self.t_steps + 1]])
            loss, h_prev, c_prev = self.training_step(inputs, outputs, np.copy(h_prev), np.copy(c_prev))

            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if iteration % 100 == 0:
                print("iter %d, loss %f\n" % (iteration, smooth_loss))
                sample_idxes = self.sample(h_prev, c_prev, inputs[0], 200)
                txt = ''.join(idx_to_char[idx] for idx in sample_idxes)
                print(txt)
                print()

            # update parameters with gradients
            for var in self.params.get_all():
                var.m += var.der * var.der
                var.val += -(self.learning_rate * var.der /np.sqrt(var.m + self.epsilon))

            iteration += 1
            index += self.t_steps

    def sample(self, h_prev, c_prev, first_char_idx, sentence_length):
        x = np.zeros((self.X_size, 1))
        x[first_char_idx] = 1

        h = h_prev
        c = c_prev

        indexes = []

        for t in range(sentence_length):
            fprop_results = self.forwardprop(h, c, x)
            h = fprop_results['h_t']
            c = fprop_results['c_t']
            idx = np.random.choice(range(self.X_size), p=fprop_results['y_hat'].ravel())
            x = np.zeros((self.X_size, 1))
            x[idx] = 1
            indexes.append(idx)

        return indexes

    # dsigmoid(logit(x)) = x * (1 - x)
    def dsigmoid_logit(self, x):
        return x * (1 - x)

    # dtanh(arctanh(x)) = 1 - x * x
    def dtanh_arctanh(self, x):
        return 1 - x * x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)


def main():
    data = open('input1.txt', 'r').read()
    chars = list(set(data))
    data_size, X_size = len(data), len(chars)
    print("data has %d characters, %d unique" % (data_size, X_size))
    lstmCell = LSTMCell(X_size)
    lstmCell.train(data, chars)

main()
