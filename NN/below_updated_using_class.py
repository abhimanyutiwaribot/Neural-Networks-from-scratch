import numpy as np     # Importing numpy

np.random.seed(0)      # the values won't change as it is initialized by 0

X =   [[1, 2, 3, 2.5],           # Inputs to the Model
       [2.0, 5.0, -1.0, 2.0],
       [-1.5, 2.7, 3.3, -0.8]]

class Layers:                   # Made a class named Layers
    def __init__(self, n_inputs, n_neurons):            # initializing the class with __init__
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)    # Getting random weights using randn and multiplying it with 0.10 (Hoping to get n_inputs between -1.0 < n_inputs < +1 ) 
        self.biases = np.zeros((1, n_neurons))      # ???? Still don't know

    def forward(self, inputs):    # Forwarding the inputs 
        self.output = np.dot(inputs, self.weights)+ self.biases   # outputing it using the weights,and biases

layer1 = Layers(4, 5)
layer2 = Layers(5, 2)
layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)

