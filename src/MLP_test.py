import MLP
import numpy as np

xorgate = MLP.MultiLayerPerceptron([2, 2, 1])  # Create a simple MLP with 2 inputs, 2 hidden neurons, and 1 output neuron
weights = [
  [], # Empty list for the input layer (no weights)
  [ [-10,-10,15], [15,15,-10] ], # Weights for the two hidden neurons (each has 2 inputs + bias)
  [ [10,10,-15] ] # Weights for the output neuron (which has 2 inputs from the hidden layer + bias)
]
xorgate.set_weights(weights)  # Set the weights for the MLP

def test_xorgate11():
  assert np.isclose(xorgate.run([1.0, 1.0]),0,atol=0.1)  # Should return a value close to 0.0
def test_xorgate10():
  assert np.isclose(xorgate.run([1.0, 0.0]),1,atol=0.1)  # Should return a value close to 1.0
def test_xorgate01():
  assert np.isclose(xorgate.run([0.0, 1.0]),1,atol=0.1)  # Should return a value close to 1.0
def test_xorgate00():
  assert np.isclose(xorgate.run([0.0, 0.0]),0,atol=0.1)  # Should return a value close to 0.0
