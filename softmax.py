import torch
import pytest
# Had to change the implementation for cases of larger values, it will go to inf, so now we find the maximum values among the input and now the exp values will be <= 1 for sure and then subtract from all to get stable exponents

def my_softmax(x, dim =-1):
  x_max = torch.max(x, dim=dim, keepdim=True).values
  x_new = x-x_max
  exp_x = torch.exp(x_new)
  sum_exp = torch.sum(exp_x, dim = dim,keepdim=True)
  return exp_x/sum_exp

def test_softmax_1d():
    x = torch.tensor([1.0, 2.0, 3.0])
    out = my_softmax(x)
    expected = torch.softmax(x, dim=0)
    assert torch.allclose(out, expected)

def test_softmax_2d():
    x = torch.tensor([[1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]])
    out = my_softmax(x, dim=1)
    expected = torch.softmax(x, dim=1)
    assert torch.allclose(out, expected)

def test_softmax_sum_to_one():
    x = torch.randn(5)
    out = my_softmax(x)
    assert torch.allclose(out.sum(), torch.tensor(1.0))

def test_softmax_negative_values():
    x = torch.tensor([-1.0, -2.0, -3.0])
    out = my_softmax(x)
    expected = torch.softmax(x, dim=0)
    assert torch.allclose(out, expected)

def test_softmax_large_values():
    x = torch.tensor([1000.0, 1001.0, 1002.0])
    out = my_softmax(x)
    expected = torch.softmax(x, dim=0)
    assert torch.allclose(out, expected)

def test_softmax_dim0():
    x = torch.randn(3, 4)
    out = my_softmax(x, dim=0)
    expected = torch.softmax(x, dim=0)
    assert torch.allclose(out, expected)

def test_softmax_dim1():
    x = torch.randn(3, 4)
    out = my_softmax(x, dim=1)
    expected = torch.softmax(x, dim=1)
    assert torch.allclose(out, expected)

def test_softmax_single_element():
    x = torch.tensor([5.0])
    out = my_softmax(x)
    assert torch.allclose(out, torch.tensor([1.0]))
