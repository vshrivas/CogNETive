import numpy as np
from NeuralLayer import NeuralLayer
import math

class NeuralNet:
    def __init__(self, layer_sizes, activation="sigmoid", loss="squared"):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes # layers is a list of the number of neurons in each layer
        self.activation_type = activation
        self.loss = loss

        # create layers based on sizes passed in
        self.layers = []
        self.layer_errors = []
        for l_size in layer_sizes:
            layer = NeuralLayer(l_size)
            self.layers.append(layer)
            self.layer_errors.append(np.empty([l_size, 1]))

        # create weights arrays
        self.weights = [] # list of numpy arrays for weights
        self.weights.append(-1) # dummy value appended, to allow one-indexing for weights
        for i in range(1, len(self.layers)):
            weight_arr = np.random.rand(self.layers[i-1].size, self.layers[i].size) # double check array size
            self.weights.append(weight_arr)

    # runs backprop alg on all training examples
    # data_pts: list of numpy arrays, where each array is a single point
    # num_epochs: number of epochs to train this for
    def train(self, data_pts, results, num_epochs):
        '''for each example:
            train the neural net for this example
            run backprop alg to recompute the weights'''
        for epoch in range(0, num_epochs):
            print("epoch: ", epoch)
            # TODO: randomly sort all data_pts
            index = 0
            for pt in data_pts:
                print("pt: ", pt)
                #print(pt.shape)
                # use the weights and activations to calculate prediction
                self.feedforward(pt)
                # improve weights based on cost
                self.backprop(results[index])
                index += 1

    # generates a prediction (y_hat) based on the input, weights, and activations
    def feedforward(self, data_pt):
        # set the data_pt as the first layer
        self.layers[0].layer = data_pt.reshape(self.layer_sizes[0], -1)
        print(self.layers[0].layer.shape)
        print("layer: ", self.layers[0].layer)
        for i in range(1, self.num_layers): # compute values at each layer
            # self.layers[i-1].layer -> (2, 1).T -> (1, 2)
            # self.weights[i] ->  (2, 3)
            # p1 has dims -> (1, 3)
            # p2 has dims -> (3, 1)
            p1 = np.dot(np.transpose(self.layers[i-1].layer), self.weights[i])
            p2 = np.transpose(p1)
            self.layers[i].layer = self.activation(p2)
            print("layer: ", self.layers[i].layer)

    # uses the backpropogation algorithm to update the weights
    def backprop(self, result):
        # calculate last layer error
        # Eo = C'(y_hat) * R'(Zo)
        self.layer_errors[self.num_layers - 1] = self.loss_der(result) * self.act_der(self.layers[self.num_layers - 1].layer)

        # calculate layer errors: Eh = E(h+1) * W(h+1) * R'(Z(h)), using backprop + memoization
        # W(h+1) -> [lh, lh+1]
        # E(h+1) -> [lh+1, 1]
        for i in reversed(range(0, self.num_layers - 1)):
            self.layer_errors[i] = np.dot(self.weights[i + 1], self.layer_errors[i + 1]) * self.act_der(self.layers[i].layer)

        # recalculate weights in each layer
        # layers[i-1] -> [l(i-1), 1]
        # layer errors[i] -> [l(i), 1]
        for i in range(1, self.num_layers):
            # update weights by layer values of i-1th layer and layer errors of ith layer
            self.weights[i] -= np.dot(self.layers[i-1].layer, np.transpose(self.layer_errors[i]))

    # evaluates the derivative of the loss function
    def loss_der(self, result):
        if self.loss == "squared":
            # C = (y_hat - y)^2
            # C' = 2(y_hat - y)
            final_layer_size = self.layer_sizes[len(self.layer_sizes) - 1]
            loss_der_val = np.empty([final_layer_size, 1])
            print("last layer: ", self.layers[self.num_layers - 1])
            print("result: ", result)
            for i in range(0, final_layer_size):
                loss_der_val[i][0] = 2 * (self.layers[self.num_layers - 1].layer[i][0] - result)

            return loss_der_val

    # evaluates the derivative of the activation function for the passed in matrix
    def act_der(self, input):
        if self.activation_type == "sigmoid":
            # S'(z) = S(z) * (1-S(z))
            return input * (1 - input)

    def activation(self, values):
        # for each row
        activated = np.empty([values.shape[0], 1])
        for i in range(0, values.shape[0]):
            if self.activation_type == "sigmoid":
                activated[i][0] = 1/(1 + math.exp(-values[i][0]))

        return activated
