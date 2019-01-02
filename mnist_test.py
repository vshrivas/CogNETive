from NeuralNet import NeuralNet
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_labels.shape)
print(test_labels.shape)
'''for i in range(0, test_labels.shape[0]):
    print(type(test_labels[0]))'''
print(type(train_images))

# scale pixel values to a range from 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

training_data = train_images.reshape(train_images.shape[0], -1) # flatten
training_data_res = np.concatenate((training_data, train_labels.reshape(train_labels.shape[0], -1)), axis=1)

test_data = test_images.reshape(test_images.shape[0], -1)
#test_data = np.concatenate((test_data, test_labels.reshape(test_labels.shape[0], -1)), axis=1)

'''training_data = np.empty((0, 784)) # flattened 28 * 28 array
for image in train_images:
    #print(image.flatten().reshape(-1, image.flatten().shape[0]).shape)
    flat_image = image.flatten()
    training_data = np.vstack((training_data, flat_image))
    print(training_data.shape)'''

print(training_data.shape)
print(test_data.shape)

NNet = NeuralNet([784, 30, 10])
NNet.train(training_data_res, 30, 10)

numMatchTest = 0.0
for i in range(0, test_data.shape[0]):
    yhat = NNet.predict(test_data[i])
    if int(yhat) == int(test_labels[i]):
        numMatchTest += 1

numMatchTrain = 0.0
for i in range(0, training_data.shape[0]):
    yhat = NNet.predict(training_data[i])
    if int(yhat) == int(train_labels[i]):
        numMatchTrain += 1
    #else:
        #print(yhat, train_labels[i])

print("Test Accuracy: ", numMatchTest/test_images.shape[0], (test_images.shape[0] - numMatchTest), " misclassified")
print("Train Accuracy: ", numMatchTrain/train_images.shape[0], (train_images.shape[0] - numMatchTrain), " misclassified")
