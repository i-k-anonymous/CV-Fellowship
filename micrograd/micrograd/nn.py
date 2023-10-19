import random
import numpy as np # just for typing!
from micrograd.engine import Value
from abc import ABC, abstractmethod

# Base class for neural network modules
class Module(ABC):
  def zero_grad(self):
    for p in self.parameters(): p.grad = 0
  
  @abstractmethod
  def parameters(self) -> list[Value]: return []

# Single Neuron class
class Neuron(Module):
  def __init__(self, num_inputs: float, non_linear=True):
    self.weights: list[Value] = ... # TODO (remember to initialize with random.uniform(-1,1)!)
    self.bias: Value = ... # TODO (initialize at 0)
    self.non_linear: bool = non_linear
  
  # process input x to neuron, return activation
  def __call__(self, x: Value | float) -> Value:
    pass # TODO (NB: Remember to use ReLU if non-linear!)

  # returns list of all parameters
  def parameters(self) -> list[Value]:
    return self.weights + [self.bias]
  
  def __repr__(self):
    return f"{'ReLU' if self.non_linear else 'Linear'}Neuron({len(self.weights)})"

# Layer of neurons
class Layer(Module):
  def __init__(self, num_inputs: float, num_outputs: float, **kwargs):
    self.neurons: list[Neuron] = ... # TODO

  # process input x to layer, return activations of each neuron (in a list)
  def __call__(self, x: list[Value | float]) -> list[Value]:
    pass # TODO

  # returns list of all parameters
  def parameters(self) -> list[Value]:
    return [p for n in self.neurons for p in n.parameters()]
  
  def __repr__(self):
    return f"Layer of {len(self.neurons)} {self.neurons[0]}"

# Multi-layer perceptron (Fully connected, feedforward neural network)
class MLP(Module):
  def __init__(self, num_inputs: float, layer_dims: list[float]):
    self.layers: list[Layer] = ... # TODO (remember that the final layer is linear!)
  
  # process input x to entire MLP, return final layer activations
  def __call__(self, x: list[Value | float] | np.ndarray) -> list[Value]:
    pass # TODO

  def parameters(self) -> list[Value]:
    return [p for layer in self.layers for p in layer.parameters()]
  
  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"