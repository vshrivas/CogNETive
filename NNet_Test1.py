from NeuralNet import NeuralNet
from NeuralLayer import NeuralLayer
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Generate a dataset and plot it
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
print(type(X))
print(type(y))
print(X.shape)
print(X[0].shape)
print(y.shape)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()

NNet = NeuralNet([2, 3, 2]) # neural net of 2 neurons, 3 neurons, 2 neurons
NNet.train(X, y, 1)
