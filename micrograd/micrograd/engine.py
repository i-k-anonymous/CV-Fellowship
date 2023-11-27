from __future__ import annotations
import math

class Value:
  """ stores a single scalar value and its gradient """

  def __init__(self, data: float, _children:set=(), _op:str=''):
    self.data: float = data
    self.grad: float = 0
    # internal variables for autograd
    self._backward, self._prev, self._op = lambda: None, set(_children), _op
  
  def __add__(self, other: Value | float):
    other: Value = other if isinstance(other, Value) else Value(other) # convert to Value
    out: Value = Value(self.data + other.data, (self, other), "+")

    # chain rule!
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward

    return out

  def __mul__(self, other: Value | float):
    other: Value = other if isinstance(other, Value) else Value(other) # convert to Value
    out: Value = Value(self.data * other.data, (self, other), "*")

    # chain rule!
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out
  
  def __pow__(self, other: float):
    assert isinstance(other, (int, float)), "only supports int/float powers"
    out: Value = Value(self.data ** other, (self,), f"**{other}")

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out
  
  # See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  # Note that we define the derivative of ReLU at 0 to be 0
  def relu(self):
    out: Value = (0 if self.data < 0 else self.data, (self,), "ReLU"

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out
  
  # returns e to the power self
  def exp(self):
    out: Value = Value(math.e ** self.data, (self,), "exp")

    def _backward():
      self.grad += (math.e ** self.data) * out.grad
    out._backward = _backward

    return out
  
  # returns the log (base e) of self
  def log(self):
    out: Value = ... # TODO

    def _backward():
      pass # TODO
    out._backward = _backward

    return out
  
  def backward(self):
    # First, topo sort all children in the graph
    # See https://en.wikipedia.org/wiki/Topological_sorting
    topo = []
    # TODO

    self.grad = 1 # ie. dy/dy = 1
    for v in reversed(topo): v._backward()
  
  # The rest of the dunder methods are implemented for convenience.
  def __neg__(self): # -self
    return self * -1

  def __radd__(self, other: Value | float): # other + self
    return self + other

  def __sub__(self, other: Value | float): # self - other
    return self + (-other)

  def __rsub__(self, other: Value | float): # other - self
    return other + (-self)

  def __rmul__(self, other: Value | float): # other * self
    return self * other

  def __truediv__(self, other: Value | float): # self / other
    return self * other**-1

  def __rtruediv__(self, other: Value | float): # other / self
    return other * self**-1

  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
