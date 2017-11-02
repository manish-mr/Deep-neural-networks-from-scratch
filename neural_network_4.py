####################################
# Neural Network 4 (MNIST database)
# Input neurons: 784
# Hidden layer 1 neurons: 300
# Hidden layer 2 neurons: 100
# Output neurons: 10
# Training on MNIST database
####################################

from NeuralNetwork import NeuralNetwork
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import time

start_time = time.time()

# Converts y containg integer values to one-hot vector values
def build_y(y):
    result = np.array([])
    for y_val in y:
        new_y = np.zeros(10)
        new_y[y_val] = 1
        if(result.shape[0] == 0):    # if result is empty
            result = new_y
        else:
            result = np.vstack((result, new_y))
    return result

# Training data
mnist = input_data.read_data_sets("/tmp/data/") # or wherever you want to put your data

X_train = mnist.train.images
y_train = mnist.train.labels.astype("int")

X_test = mnist.test.images
y_test = mnist.test.labels.astype("int")

# Configure network
n_inputs = 784
hidden_layers = [300, 100] # 2 hidden layers
n_output = 10
learning_rate = 0.016
momentum = 0.8
np.random.seed(1)
nn = NeuralNetwork(n_inputs, hidden_layers, n_output, learning_rate, momentum)
    
plt_x_vals =[]
plt_y_vals = []

# Training
epochs = 40
batch_size = 100
y_train = build_y(y_train)
for epoch in range(epochs):
    plt_x_vals.append(epoch)
    error = nn.train(X_train, y_train, batch_size, False, nn.relu, nn.relu_deriv, nn.sigmoid, nn.sigmoid_deriv)
    MSE = np.mean(error*error)
    plt_y_vals.append(MSE)
    print("Epoch: ", epoch, ", MSE: ", MSE)
        
print("Time required for Training: ","%s seconds" % (time.time() - start_time))

# Testing
prediction = nn.predict(X_test, nn.relu, nn.sigmoid)
match_count = 0
total_test_count = y_test.shape[0]
y_test = build_y(y_test)

for i in range(total_test_count):
    if(np.array_equal(y_test[i], prediction[i])):
        match_count += 1
    
print("Test performance: ", match_count/total_test_count)

# MSE Plot
plt.plot(plt_x_vals, plt_y_vals, linestyle='solid')
plt.title("MSE plot")
plt.ylabel("MSE")
plt.xlabel("Epochs")
plt.show()