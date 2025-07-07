import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = np.random.rand(inputs + 1) * 2 + 1  # Random weights for inputs and bias
        self.bias = bias
    
    def set_weights(self, weights):
        """Set the weights of the perceptron. The length of the weights must be inputs + 1 (including bias)."""
        if len(weights) != len(self.weights):
            raise ValueError("Weights length must match number of inputs + 1 (including bias).")
        self.weights = np.array(weights)

    def sigmoid(self, x):
        """Sigmoid activation function.""" 
        return 1 / (1 + np.exp(-x))

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(x_sum)
