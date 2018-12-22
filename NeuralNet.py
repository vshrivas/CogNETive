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

    # generates a prediction based on the input
    def feedforward(self):
        pass

    def train(self):

    # uses the backpropogation algorithm to update the weights
    def backprop(self):
        # calculate last layer error
        # Eo = C'(y_hat) * R'(Zo)
        self.layer_errors[self.num_layers - 1] = self.loss_der() * self.act_der(self.layer_inputs(self.num_layers - 1))

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

    # finds the derivative of the loss function
    def loss_der(self):
        pass

    # evaluates the derivative of the activation function for the passed in matrix
    def act_der(self, input):
        pass

    def layer_inputs(self, layer_index):
        pass
