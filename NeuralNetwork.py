import numpy as np

# Neural Network class includes functions for initializing, training and testing
# your neural network
class NeuralNetwork:
    
    # Generate random matrix
    def generate_matrix(self, x, y):
        return np.random.normal(scale=0.1, size=(x, y))
    
    # Sigmoid function
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x*(1.0-x)
    
    # ReLU function
    def relu(self, x):
        return np.maximum(x, 0)
        
    def relu_deriv(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    # Softmax function
    def softmax(x):
        assert len(x.shape) == 2
        s = np.max(x, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(x - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div

    def softmax_deriv(x):
        return x*(1.0-x)

    # Configure network
    # n_input & n_output denotes the count of neurons at input and output layers
    # hidden_layers is an array of count of neurons in each hidden layer
    # For ex: If hidden_layers = [5,10], then it means there
    # are 2 hidden layers in this network with 5 and 10 neurons respectively
    # len(hidden_layers) should be atleast one
    def __init__(self, n_input, hidden_layers, n_output, learning_rate, momentum):
        self.n_input = n_input
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.hidden_layers_count = len(hidden_layers)
        self.weights = []
        self.biases = []
        
        # initializing first weight & bias
        W_first = self.generate_matrix(n_input, hidden_layers[0])
        b_first = self.generate_matrix(1, hidden_layers[0])
        self.weights.append(W_first)
        self.biases.append(b_first)
        
        # initialize all middle weights & biases
        if(self.hidden_layers_count > 1):
            last_num = 0
            for num in hidden_layers:
                if(last_num != 0):
                    weight = self.generate_matrix(last_num, num)
                    bias = self.generate_matrix(1, num)
                    self.weights.append(weight)
                    self.biases.append(bias)
                last_num = num
                
        # initializing last weight & bias
        W_last = self.generate_matrix(hidden_layers[len(hidden_layers)-1], n_output)
        b_last = self.generate_matrix(1, n_output)
        self.weights.append(W_last)
        self.biases.append(b_last)
        

    # Train the network with the specified configuration. This function always
    # trains for 1 epoch, so calling routine has to handle epochs for training
    # X: input
    # y: actual output
    # batch size: the batch size at which you want to train the netwprk
    # use_momentum: True or False, depending upon whether you want to use
    # momentum learning or not while training weights. If True, then it will
    # use configured momentum value
    # hidden_fn: hidden function to use. Ex: ReLU
    # hidden_deriv: hidden derivative funtion
    # output_fn: output function to use. Ex: Sigmoid, Softmax
    # output_deriv: output dervative function
    # returns: output error
    def train(self, X_train, y_train, batch_size, use_momentum, hidden_fn, hidden_deriv, output_fn, output_deriv):
        intervals = int(y_train.shape[0]/batch_size)
        
        for i in range(intervals):
            start_index = i * batch_size
            end_index = start_index + batch_size
            X = X_train[start_index:end_index,:]
            y = y_train[start_index:end_index]
            
            ### forward propogate ###
            h_output = []
            # process dot products
            weights_count = len(self.weights)
            for i in range(weights_count):
                if(i == 0): #first
                    h = hidden_fn(np.dot(X,self.weights[i]) + self.biases[i])
                    h_output.append(h)
                elif(i == (weights_count-1)):   #last element
                    h = output_fn(np.dot(h_output[i-1],self.weights[i]) + self.biases[i])
                    h_output.append(h)
                else:
                    h = hidden_fn(np.dot(h_output[i-1],self.weights[i]) + self.biases[i])
                    h_output.append(h)
                    
            ### back propogation ###
            i = 0
            j = len(self.weights) - 1   # index for traversing weights
            first_h_processed = False
            deltas = []     # deltas will get stored in reverse order
            for h in reversed(h_output):
                if(not first_h_processed):
                    error = y - h
                    E = error
                    delta = error*output_deriv(h)
                    deltas.append(delta)
                    first_h_processed = True
                else:
                    error = np.dot(deltas[i], self.weights[j].T)
                    delta = error*hidden_deriv(h)
                    deltas.append(delta)
                    i = i + 1
                    j = j - 1
                    
            ### update weights ###
            i = 0   # index for traversing deltas
            j = len(h_output) - 2
            k = len(self.biases) - 1
            for w in reversed(range(len(self.weights))):
                # for last element of reversed weights array
                if(w == 0):
                    if(use_momentum):
                        self.weights[w] = self.momentum*self.weights[w] + (self.learning_rate * (X.T.dot(deltas[i])))
                    else:
                        self.weights[w] = self.weights[w] + (self.learning_rate * (X.T.dot(deltas[i])))
                    self.biases[k] = self.biases[k] + np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate
                else:   # for all other elements
                    if(use_momentum):
                        self.weights[w] = self.momentum*self.weights[w] + (self.learning_rate * (h_output[j].T.dot(deltas[i])))
                    else:
                        self.weights[w] = self.weights[w] + (self.learning_rate * (h_output[j].T.dot(deltas[i])))
                    self.biases[k] = self.biases[k] + np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate
                    i = i + 1
                    j = j - 1
                    k = k - 1
        
        return E
        
    # Predict output for X
    # This function should be called once weights are trained
    def predict(self, X, hidden_fn, output_fn):
        ### forward propogate ###
        h_output = []
        # process dot products
        weights_count = len(self.weights)
        for i in range(weights_count):
            if(i == 0): #first
                h = hidden_fn(np.dot(X,self.weights[i]) + self.biases[i])
                h_output.append(h)
            elif(i == (weights_count-1)):   #last element
                h = output_fn(np.dot(h_output[i-1],self.weights[i]) + self.biases[i])
                h_output.append(h)
            else:
                h = hidden_fn(np.dot(h_output[i-1],self.weights[i]) + self.biases[i])
                h_output.append(h)
        
        out = h_output[len(h_output)-1]
        return (out > 0.5).astype(int)
