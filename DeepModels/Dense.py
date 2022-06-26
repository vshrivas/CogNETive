from Layer import Layer

class Dense(Layer):
    def __init__(self, input_dim, output_dim, activation):
        self.W = None # create weight matrix of dims (input_dim, output_dim)
        self.b = None # create bias array (output_dim, 1)
        self.Z = None # create array for Z
        self.A = None # create array for A

        self.dZ = None # dZ array
        self.dA = None # dA array
        self.dW = None
        self.db = None

        self.activation = activation 
        pass

    # run forward prop on a batch of examples
    def forward(self, x):
        self.A_prev = x
        # Z = weights * self.A_prev + bias
        # A = activation(Z)
        # return A
        pass

    def backprop_step(self, cache_next):
        ''' cache_next contains info from next layer in network '''
        self.dA = cache_next['dA']
        self.dZ = self.dA * self.activation.der(self.Z)
        self.dW += self.dZ * self.A_prev
        self.db += self.dZ

        self.dA_prev = self.dZ * self.W 
        cache_curr = {'dA':self.dA_prev}
        return cache_curr
