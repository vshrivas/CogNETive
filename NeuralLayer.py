import numpy as np
class NeuralLayer:
    def __init__(self, size):
        self.size = size
        self.layer = np.random.rand(size, 1)
