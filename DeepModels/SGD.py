class SGD(object):
    def __init__(self, layers, lr): # params: [{'w':weights, 'b':biases}] ordered by layer, learning_rate
        self.layers = layers
        self.lr = lr

    def backprop_step(self, cache):
        # perform backprop across layers (accumulates gradients) (should the model do this instead, if it doesn't depend on optimizer)
        for layer in reversed(self.layers):
            cache_next = layer.backprop_step(cache)
            cache = cache_next

        # update weights and biases of layers (W -= (lr/batch_size)(dW))



