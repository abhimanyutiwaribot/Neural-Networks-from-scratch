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

layer1 = Layers(2, 5)
layer1.forward(X)
activation1 = Activation_ReLU()
activation1.forward(layer1.output)

print(activation1.output)




