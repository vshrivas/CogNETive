import numpy as np
from NeuralLayer import NeuralLayer
import math
import tensorflow as tf

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
            self.layer_errors.append(np.zeros([l_size, 1]))

        # create weights arrays
        self.weights = [] # list of numpy arrays for weights
        self.weights.append(-1) # dummy value appended, to allow one-indexing for weights
        for i in range(1, len(self.layers)):
            weight_arr = np.random.randn(self.layers[i-1].size, self.layers[i].size) # double check array size
            self.weights.append(weight_arr)

        self.epsilon = 3

    # runs backprop alg on all training examples
    # data_pts: list of numpy arrays, where each array is a single point
    # num_epochs: number of epochs to train this for
    def train(self, input, num_epochs):
        '''for each example:
            train the neural net for this example
            run backprop alg to recompute the weights'''
        for epoch in range(0, num_epochs):
            #print("epoch: ", epoch)
            np.random.shuffle(input)
            # TODO: randomly sort all data_pts
            avg_loss = 0
            data_pts = input[...,0:input.shape[1]-1]
            #print(data_pts.shape)
            results = input[...,input.shape[1]-1]
            #print(results.shape)
            for i in range(0, data_pts.shape[0]):
                #print("pt: ", pt)
                #print(pt.shape)
                # use the weights and activations to calculate prediction
                self.feedforward(data_pts[i], epoch)

                '''result = np.zeros([self.layer_sizes[self.num_layers - 1], 1]) # zero probablities for all categories, except correct one
                result[int(results[i])] = 1 # 100% probablity of correct category
                loss = self.find_loss(result)
                #break
                avg_loss += loss'''
                # improve weights based on cost
                self.backprop(results[i], epoch)

                #if i == 10:
                    #break

            numMatch = 0
            for i in range(0, 10000):
                yhat = self.predict(data_pts[i])
                if int(yhat) == int(results[i]):
                    numMatch += 1
            #break
            print("accuracy epoch ", epoch, ":", numMatch, " / 10000")
            #break

    # generates an output (y_hat) based on the input, weights, and activations
    def feedforward(self, data_pt, epoch):
        # set the data_pt as the first layer
        self.layers[0].layer = data_pt.reshape(self.layer_sizes[0], -1)
        #print("layer0", self.layers[0].layer)
        #print(self.layers[0].layer.shape)
        #print("layer: ", self.layers[0].layer)
        for i in range(1, self.num_layers): # compute values at each layer
            # self.layers[i-1].layer -> (2, 1).T -> (1, 2)
            # self.weights[i] ->  (2, 3)
            # p1 has dims -> (1, 3)
            # p2 has dims -> (3, 1)
            #print("weights", i, self.weights[i])
            p1 = np.matmul(np.transpose(self.layers[i-1].layer), self.weights[i])
            p2 = np.transpose(p1)
            self.layers[i].layer = self.activation(p2)
            '''if epoch >= 0:
                print("weights", self.weights[i])
                print("before activation layer ", i, p2)
                print("after activation layer ", i, self.layers[i].layer)'''
            #print("layer ", i, self.layers[i].layer)
            #print("layer: ", self.layers[i].layer)
            #print("\n")

        #self.layers[self.num_layers - 1].layer = self.softmax(self.layers[self.num_layers - 1].layer)

    # generates predictions on the input data, should be used after training network
    def predict(self, data_pt):
        self.feedforward(data_pt, 1)
        #prediction_probs = self.softmax(self.layers[self.num_layers - 1].layer)
        # find largest prob in neurons
        largest_prob_neuron = 0
        for i in range(0, self.layer_sizes[self.num_layers - 1]):
            if self.layers[self.num_layers - 1].layer[i][0] > self.layers[self.num_layers - 1].layer[largest_prob_neuron][0]:
                largest_prob_neuron = i

        return largest_prob_neuron

    # uses the backpropogation algorithm to update the weights
    def backprop(self, y_val, epoch):
        #print("y_val", y_val)
        # generate what the actual result vector should be
        result = np.zeros([self.layer_sizes[self.num_layers - 1], 1]) # zero probablities for all categories, except correct one
        result[int(y_val)][0] = 1 # 100% probablity of correct category

        # calculate last layer error
        # Eo = C'(y_hat) * R'(Zo)
        self.layer_errors[self.num_layers - 1] = self.loss_der(result) * self.act_der(self.layers[self.num_layers - 1].layer)
        '''if epoch > 1:
            print("loss der:", self.loss_der(result))
            print("act der:", self.act_der(self.layers[self.num_layers - 1].layer))
            print("last layer error: ", self.layer_errors[self.num_layers - 1])'''
        # calculate layer errors: Eh = E(h+1) * W(h+1) * R'(Z(h)), using backprop + memoization
        # W(h+1) -> [lh, lh+1]
        # E(h+1) -> [lh+1, 1]Sure
        #print("start")
        for i in reversed(range(0, self.num_layers - 1)):
            #print(i)
            '''print("act der layer ", i, self.act_der(self.layers[i].layer))
            print("layer ", i, self.layers[i].layer)'''
            self.layer_errors[i] = np.dot(self.weights[i + 1], self.layer_errors[i + 1]) * self.act_der(self.layers[i].layer)

        # recalculate weights in each layer
        # layers[i-1] -> [l(i-1), 1]
        # layer errors[i] -> [l(i), 1]
        for i in range(1, self.num_layers):
            # update weights by layer values of i-1th layer and layer errors of ith layer
            self.weights[i] -= self.epsilon * np.dot(self.layers[i-1].layer, np.transpose(self.layer_errors[i]))
            '''if epoch > 1:
                print("prev layer: ", self.layers[i-1].layer)
                print("layer errors:", np.transpose(self.layer_errors[i]))
                print("delta of weights ", i, self.epsilon * np.dot(self.layers[i-1].layer, np.transpose(self.layer_errors[i])))'''

    def softmax(self, result):
        return np.exp(result)/np.sum(np.exp(result))

    def find_loss(self, result):
        #print("last layer: ", self.layers[self.num_layers - 1].layer)
        #print("res: ", result)
        if self.loss == "squared":
            #softmax = self.softmax(self.layers[self.num_layers - 1].layer)
            return np.sum((self.layers[self.num_layers - 1].layer - result) ** 2)

        elif self.loss == "cross_entropy":
            #softmax = self.softmax(self.layers[self.num_layers - 1].layer)
            return -np.sum(result * np.log(softmax))

    # evaluates the derivative of the loss function
    def loss_der(self, result):
        if self.loss == "squared":
            # C = (y_hat - y)^2
            # C' = 2(y_hat - y)
            '''final_layer_size = self.layer_sizes[len(self.layer_sizes) - 1]
            loss_der_val = np.empty([final_layer_size, 1])
            #print("last layer: ", self.layers[self.num_layers - 1])
            #print("result: ", result)
            for i in range(0, final_layer_size):
                loss_der_val[i][0] = 2 * (self.layers[self.num_layers - 1].layer[i][0] - result[i][0])

            return loss_der_val'''
            #softmax = self.softmax(self.layers[self.num_layers - 1].layer)
            return 2 * (self.layers[self.num_layers - 1].layer - result)

        elif self.loss == "cross_entropy":
            #softmax = self.softmax(self.layers[self.num_layers - 1].layer)
            return -np.sum(result / self.layers[self.num_layers - 1].layer)

    # evaluates the derivative of the activation function for the passed in matrix
    def act_der(self, input):
        if self.activation_type == "sigmoid":
            # S'(z) = S(z) * (1-S(z))
            return input * (1 - input)

    def activation(self, values):
        values = np.clip(values, -500, 500)
        return 1/(1 + np.exp(-values))
        """# for each row
        activated = np.empty([values.shape[0], 1])
        for i in range(0, values.shape[0]):
            if self.activation_type == "sigmoid":
                print(-values[i][0])
                activated[i][0] = 1/(1 + math.exp(-values[i][0]))

        return activated"""
