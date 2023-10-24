import torch
from micrograd.engine import Value

tol = 1e-6

def test_add():
  xmg = Value(1.0)
  ymg = xmg + Value(2.0)
  ymg.backward()

  xpt = torch.Tensor([1.0])
  xpt.requires_grad = True
  ypt = xpt + torch.Tensor([2.0])
  ypt.backward()

  assert ymg.data == ypt.data.item(), "forward pass went wrong"
  assert xmg.grad == xpt.grad.item(), "backward pass went wrong"

def test_mul():
  xmg = Value(4.0)
  ymg = xmg * Value(2.0)
  ymg.backward()

  xpt = torch.Tensor([4.0])
  xpt.requires_grad = True
  ypt = xpt * torch.Tensor([2.0])
  ypt.backward()

  assert ymg.data == ypt.data.item(), "forward pass went wrong"
  assert xmg.grad == xpt.grad.item(), "backward pass went wrong"

def test_pow():
  xmg = Value(-4.0)
  ymg = xmg**3
  ymg.backward()

  xpt = torch.Tensor([-4.0])
  xpt.requires_grad = True
  ypt = xpt**3
  ypt.backward()

  assert ymg.data == ypt.data.item(), "forward pass went wrong"
  assert xmg.grad == xpt.grad.item(), "backward pass went wrong"

def test_relu_neg():
  xmg = Value(-4.0)
  ymg = xmg.relu()
  ymg.backward()

  xpt = torch.Tensor([-4.0])
  xpt.requires_grad = True
  ypt = xpt.relu()
  ypt.backward()

  assert ymg.data == ypt.data.item(), "forward pass went wrong"
  assert xmg.grad == xpt.grad.item(), "backward pass went wrong"

def test_relu_pos():
  xmg = Value(4.0)
  ymg = xmg.relu()
  ymg.backward()

  xpt = torch.Tensor([4.0])
  xpt.requires_grad = True
  ypt = xpt.relu()
  ypt.backward()

  assert ymg.data == ypt.data.item(), "forward pass went wrong"
  assert xmg.grad == xpt.grad.item(), "backward pass went wrong"

def test_relu_zero():
  xmg = Value(0.0)
  ymg = xmg.relu()
  ymg.backward()

  xpt = torch.Tensor([0.0])
  xpt.requires_grad = True
  ypt = xpt.relu()
  ypt.backward()

  assert ymg.data == ypt.data.item(), "forward pass went wrong"
  assert xmg.grad == xpt.grad.item(), "backward pass went wrong"

def test_exp():
  xmg = Value(1.0)
  ymg = xmg.exp()
  ymg.backward()

  xpt = torch.Tensor([1.0])
  xpt.requires_grad = True
  ypt = torch.exp(xpt)
  ypt.backward()

  assert abs(ymg.data - ypt.data.item()) < tol, "forward pass went wrong"
  assert abs(xmg.grad - xpt.grad.item()) < tol, "backward pass went wrong"

def test_log():
  xmg = Value(1.0)
  ymg = xmg.log()
  ymg.backward()

  xpt = torch.Tensor([1.0])
  xpt.requires_grad = True
  ypt = torch.log(xpt)
  ypt.backward()

  assert ymg.data == ypt.data.item(), "forward pass went wrong"
  assert xmg.grad == xpt.grad.item(), "backward pass went wrong"

def test_sanity_check():

  x = Value(-4.0)
  z = 2 * x + 2 + x
  q = z.relu() + z * x
  h = (z * z).relu()
  y = h + q + q * x
  y.backward()
  xmg, ymg = x, y

  x = torch.Tensor([-4.0]).double()
  x.requires_grad = True
  z = 2 * x + 2 + x
  q = z.relu() + z * x
  h = (z * z).relu()
  y = h + q + q * x
  y.backward()
  xpt, ypt = x, y

  # forward pass went well
  assert ymg.data == ypt.data.item()
  # backward pass went well
  assert xmg.grad == xpt.grad.item()

def test_more_ops():

  a = Value(-4.0)
  b = Value(2.0)
  c = a + b
  d = a * b + b**3
  c += c + 1
  c += 1 + c + (-a)
  d += d * 2 + (b + a).relu()
  d += 3 * d + (b - a).relu()
  e = c - d
  f = e**2
  g = f / 2.0
  g += 10.0 / f
  g.backward()
  amg, bmg, gmg = a, b, g

  a = torch.Tensor([-4.0]).double()
  b = torch.Tensor([2.0]).double()
  a.requires_grad = True
  b.requires_grad = True
  c = a + b
  d = a * b + b**3
  c = c + c + 1
  c = c + 1 + c + (-a)
  d = d + d * 2 + (b + a).relu()
  d = d + 3 * d + (b - a).relu()
  e = c - d
  f = e**2
  g = f / 2.0
  g = g + 10.0 / f
  g.backward()
  apt, bpt, gpt = a, b, g

  # forward pass went well
  assert abs(gmg.data - gpt.data.item()) < tol
  # backward pass went well
  assert abs(amg.grad - apt.grad.item()) < tol
  assert abs(bmg.grad - bpt.grad.item()) < tol