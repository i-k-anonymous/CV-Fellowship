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
    self.weights: list[Value] = [Value(random.uniform(-1,1)) for _ in range(num_inputs)]
    self.bias: Value = Value(0)
    self.non_linear: bool = non_linear
  
  # process input x to neuron, return activation
  def __call__(self, x: Value | float) -> Value:
    act = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
    return act.relu() if self.non_linear else act

  # returns list of all parameters
  def parameters(self) -> list[Value]:
    return self.weights + [self.bias]
  
  def __repr__(self):
    return f"{'ReLU' if self.non_linear else 'Linear'}Neuron({len(self.weights)})"

# Layer of neurons
class Layer(Module):
  def __init__(self, num_inputs: float, num_outputs: float, **kwargs):
    self.neurons: list[Neuron] = [Neuron(num_inputs, **kwargs) for _ in range(num_outputs)]

  # process input x to layer, return activations of each neuron (in a list)
  def __call__(self, x: list[Value | float]) -> list[Value]:
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out # NB: this squeezes for ease of use

  # returns list of all parameters
  def parameters(self) -> list[Value]:
    return [p for n in self.neurons for p in n.parameters()]
  
  def __repr__(self):
    return f"Layer of {len(self.neurons)} {self.neurons[0]}"

# Multi-layer perceptron (Fully connected, feedforward neural network)
class MLP(Module):
  def __init__(self, num_inputs: float, layer_dims: list[float]):
    size = [num_inputs] + layer_dims
    self.layers: list[Layer] = [Layer(size[i], size[i+1], non_linear=i!=len(layer_dims)-1) for i in range(len(layer_dims))]
  
  # process input x to entire MLP, return final layer activations
  def __call__(self, x: list[Value | float] | np.ndarray) -> list[Value]:
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self) -> list[Value]:
    return [p for layer in self.layers for p in layer.parameters()]
  
  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"