import MLP
import numpy as np

andgate = MLP.Perceptron(2)
andgate.set_weights([10.0, 10.0, -15.0])  # Set weights for two inputs and one bias to act as a logical AND gate
orgate = MLP.Perceptron(2)
orgate.set_weights([10.0, 10.0, -5.0])  # Set weights for two inputs and one bias to act as a logical OR gate

def test_andgate11():
  assert np.isclose(andgate.run([1.0, 1.0]),1,atol=0.1)  # Should return a value close to 1.0
def test_andgate10():
  assert np.isclose(andgate.run([1.0, 0.0]),0,atol=0.1)  # Should return a value close to 0.0
def test_andgate01():
  assert np.isclose(andgate.run([0.0, 1.0]),0,atol=0.1)  # Should return a value close to 0.0
def test_andgate00():
  assert np.isclose(andgate.run([0.0, 0.0]),0,atol=0.1)  # Should return a value close to 0.0

def test_orgate11():
  assert np.isclose(orgate.run([1.0, 1.0]),1,atol=0.1)  # Should return a value close to 1.0
def test_orgate10():
  assert np.isclose(orgate.run([1.0, 0.0]),1,atol=0.1)  # Should return a value close to 1.0
def test_orgate01():
  assert np.isclose(orgate.run([0.0, 1.0]),1,atol=0.1)  # Should return a value close to 1.0
def test_orgate00():
  assert np.isclose(orgate.run([0.0, 0.0]),0,atol=0.1)  # Should return a value close to 0.0
