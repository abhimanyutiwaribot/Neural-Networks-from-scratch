import numpy as np     # Importing numpy
import nnfs            # Importing NNFS from NNFS.IO
from nnfs.datasets import spiral_data    # data importing

nnfs.init()    # initalization of nnfs

# X =   [[1, 2, 3, 2.5],           # Inputs to the Model
#        [2.0, 5.0, -1.0, 2.0],
#        [-1.5, 2.7, 3.3, -0.8]]


X, y = spiral_data(10, 3)  # for feature data sets and their classes in this case we have 100 feature_sets and
                            # 3 classes
# print(X , y)

class Layers:                   # Made a class named Layers
    def __init__(self, n_inputs, n_neurons):            # initializing the class with __init__
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)    # Getting random weights using randn and 
                                                                      # multiplying it with 0.10 (Hoping to get n_inputs between -1.0 < n_inputs < +1 ) 
        self.biases = np.zeros((1, n_neurons))      # ???? Still don't know

    def forward(self, inputs):    # Forwarding the inputs 
        self.output = np.dot(inputs, self.weights)+ self.biases   # outputing it using the weights,and biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

layer1 = Layers(2, 3)
activation_1 = Activation_ReLU()

layer1.forward(X)
activation_1.forward(layer1.output)
# print(activation_1.output)

layer2 = Layers(3, 3)
activation_2 = Activation_Softmax()

layer2.forward(layer1.output)
activation_2.forward(layer2.output)

print(activation_2.output[0:5])   # first five





