import numpy as np
# weights = [[0.2, 0.8, -0.5, 1.0], 
#            [0.5, -0.91, 0.26, -0.5], 
#            [-0.26, -0.27, 0.17, 0.87]]

# weights = [[1, 0, 0, 1], 
#            [0, 1, 0, 2], 
#            [0, 0, 1, 1],]

# inputs =  [[0, 0, 1, 4], 
#            [0, 1, 0, 5], 
#            [1, 0, 0, 5],]

# weights(elements in (row 1)) = inputs(elements in (column 1))
# weights(no. of columns) = inputs(no. of rows)
# weights(order(3x4)) = inputs(order(1x4))
# column = row 

inputs =   [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]


# biases = [2, 3, 0.5]

# output = np.dot(inputs, np.array(weights).T) 
# print(output) 

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
print(0.10 * np.random.randn(4, 5))
