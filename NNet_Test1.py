from NeuralNet import NeuralNet
from NeuralLayer import NeuralLayer
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

numpts = 2000
# Generate a dataset and plot it
np.random.seed(0)
X, y = datasets.make_moons(numpts)
#print(type(X))
#print(type(y))
#print(X.shape)
#print(X[0].shape)
#print(y.shape)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()

NNet = NeuralNet([2, 20, 2]) # neural net of 2 neurons, 3 neurons, 2 neurons
concatenated = np.concatenate((X, y.reshape(y.shape[0], -1)), axis=1)

test_set = concatenated[1500:]
train_set = concatenated[:1500]

print(train_set.shape)
print(test_set.shape)
#print(concatenated.shape)
NNet.train(train_set, 600)

numMatch = 0.0
data_pts = test_set[...,0:2]
y_pts = test_set[...,2]

for i in range(0, 500):
    yhat = NNet.predict(data_pts[i])
    if yhat == y_pts[i]:
        numMatch += 1
    else:
        print(yhat, y_pts[i])

print(numMatch/500, (500 - numMatch))
