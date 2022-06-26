class Layer(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim 
        self.output_dim = output_dim 

    def forward(self, x):
        pass

    def backprop_step(self, cache_next):
        pass