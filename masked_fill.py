import pytest
import torch

def my_masked_fill(x, mask, value):
    value_tensor = torch.full_like(x, value)
    return mask*value_tensor + (~mask)*x

def test_basic():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    mask = torch.tensor([True, False, True, False])
    out = my_masked_fill(x, mask, 0)
    expected = torch.tensor([0., 2., 0., 4.])
    assert torch.allclose(out, expected)

def test_against_torch():
    x = torch.randn(10)
    mask = torch.rand(10) > 0.5
    value = 5.0
    out = my_masked_fill(x, mask, value)
    expected = x.masked_fill(mask, value)
    assert torch.allclose(out, expected)

def test_2d_tensor():
    x = torch.tensor([[1., 2.], [3., 4.]])
    mask = torch.tensor([[True, False], [False, True]])
    out = my_masked_fill(x, mask, -1)
    expected = torch.tensor([[-1., 2.], [3., -1.]])
    assert torch.allclose(out, expected)

def test_all_true():
    x = torch.randn(5)
    mask = torch.ones(5, dtype=torch.bool)
    out = my_masked_fill(x, mask, 9.0)
    expected = torch.full_like(x, 9.0)
    assert torch.allclose(out, expected)

def test_all_false():
    x = torch.randn(5)
    mask = torch.zeros(5, dtype=torch.bool)
    out = my_masked_fill(x, mask, 9.0)
    assert torch.allclose(out, x)

def test_negative_values():
    x = torch.tensor([-1., -2., -3.])
    mask = torch.tensor([True, False, True])
    out = my_masked_fill(x, mask, 0.0)
    expected = torch.tensor([0., -2., 0.])
    assert torch.allclose(out, expected)

def test_value_type_float_handling():
    x = torch.randn(5)
    mask = torch.tensor([True, False, True, False, True])
    out = my_masked_fill(x, mask, 3.5)
    expected = x.masked_fill(mask, 3.5)
    assert torch.allclose(out, expected)

def test_single_element():
    x = torch.tensor([5.0])
    mask = torch.tensor([True])
    out = my_masked_fill(x, mask, 1.0)
    assert torch.allclose(out, torch.tensor([1.0]))
