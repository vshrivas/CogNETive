import numpy as np

class NeuralNet:
    def __init__(self, layer_sizes, activation="sigmoid", loss="squared"):
        self.num_layers = len(self.layer_sizes)
        self.layer_sizes = layer_sizes # layers is a list of the number of neurons in each layer
        self.activation = activation
        self.loss = loss

        # create layers based on sizes passed in
        self.layers = []
        for l_size in layer_sizes:
            layer = NeuralLayer(l_size)
            self.layers.append(layer)

        # create weights arrays
        self.weights = [] # list of numpy arrays for weights
        self.weights.append(-1) # dummy value appended, to allow one-indexing for weights

        for i in range(0, len(self.layers) - 1):
            weight_arr = np.empty([self.layers[i].size, self.layers[i+1].size]) # double check array size
            self.weights.append(weight_arr)

        self.layer_errors = []

    # runs backprop alg on all training examples
    # data_pts: list of numpy arrays, where each array is a single point
    # num_epochs: number of epochs to train this for
    def train(self, data_pts, results, num_epochs):
        '''for each example:
            train the neural net for this example
            run backprop alg to recompute the weights'''
        for epoch in num_epochs:
            # randomly sort all data_pts
            index = 0
            for pt in data_pts:
                # use the weights and activations to calculate prediction
                self.feedforward(pt)
                # improve weights based on cost
                self.backprop(results[index])
                index += 1

    # generates a prediction (y_hat) based on the input, weights, and activations
    def feedforward(self, data_pt):
        # set the data_pt as the first layer
        self.layers[0].layer = data_pt
        for i in range(1, self.num_layers): # compute values at each layer
            self.layers[i].layer = self.activation(np.transpose(np.dot(np.transpose(self.layers[i-1]), self.weights[i])))

    # uses the backpropogation algorithm to update the weights
    def backprop(self, result):
        # calculate last layer error
        # Eo = C'(y_hat) * R'(Zo)
        self.layer_errors[self.num_layers - 1] = self.loss_der(result) * self.act_der(self.layer_inputs(self.num_layers - 1))

        # calculate layer errors: Eh = E(h+1) * W(h+1) * R'(Z(h)), using backprop + memoization
        # W(h+1) -> [lh, lh+1]
        # E(h+1) -> [lh+1, 1]
        for i in reversed(range(0, self.num_layers - 1)):
            self.layer_errors[i] = np.dot(self.weights[i + 1], self.layer_errors[i + 1]) * self.act_der(self.layer_inputs(i))

        # recalculate weights in each layer
        # layers[i-1] -> [l(i-1), 1]
        # layer errors[i] -> [l(i), 1]
        for i in range(0, self.num_layers):
            # update weights by layer values of i-1th layer and layer errors of ith layer
            self.weights[i] -= np.dot(self.layers[i-1], np.transpose(self.layer_errors[i]))

    # evaluates the derivative of the loss function
    def loss_der(self, result):
        if self.loss == "squared":
            # C = (y_hat - y)^2
            # C' = 2(y_hat - y)
            final_layer_size = self.layer_sizes[len(self.layer_sizes) - 1]
            loss_der_val = np.empty([final_layer_size, 1])
            for i in range(0, final_layer_size):
                loss_der_val[i][0] = 2 * (self.layers[self.num_layers - 1].layer[i][0] - result[i][0])

            return loss_der_val

    # evaluates the derivative of the activation function for the passed in matrix
    def act_der(self, input):
        pass

    # calculates the reverse of the activation values
    def layer_inputs(self, layer_index):
        pass
