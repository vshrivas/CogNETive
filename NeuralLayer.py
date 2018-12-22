import numpy as np
class NeuralLayer:
    def __init__(self, size):
        self.size = size
        self.values = np.empty([size, 1])
