import numpy as np

class Var:
    def __init__(self, name, val):
        self.name = name
        self.val = val
        self.der = np.zeros_like(val)
        self.m = np.zeros_like(val)

class Params:
    # Z, concatenation of h prev and X, has dimensions (Z x 1)
    # W * Z -> (h_size, 1) so W has dimensions ()
    def __init__(self, H_size, X_size, z_size, weight_sd):
        self.W_f = Var('W_f', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_f = Var('b_f', np.zeros((H_size, 1)))

        self.W_i = Var('W_i', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_i = Var('b_i', np.zeros((H_size, 1)))

        self.W_cbar = Var('W_Cbar', np.random.randn(H_size, z_size) * weight_sd)
        self.b_cbar = Var('b_Cbar', np.zeros((H_size, 1)))

        self.W_o = Var('W_o', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_o = Var('b_o', np.zeros((H_size, 1)))

        #For final layer to predict the next character
        self.W_v = Var('W_v', np.random.randn(X_size, H_size) * weight_sd)
        self.b_v = Var('b_v', np.zeros((X_size, 1)))

    def get_all(self):
        return [self.W_f, self.W_i, self.W_cbar, self.W_o, self.W_v,
               self.b_f, self.b_i, self.b_cbar, self.b_o, self.b_v]
