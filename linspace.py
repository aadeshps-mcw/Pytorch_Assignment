import torch
import pytest
# Had to chage my implemetation to include the error caused if steps<=1, so added coniditons to handle them
# The difference btwn arange and range is the end, arange = [start,end) but range = [start,end].

def my_linspace(start, end, steps):
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if steps == 1:
        return torch.tensor([start], dtype=torch.float32)
    step = (end - start) / (steps - 1)
    idx = torch.arange(steps, dtype=torch.float32)
    out = start + step * idx
    return out

def test_basic():
    out = my_linspace(0, 10, 5)
    expected = torch.tensor([0., 2.5, 5., 7.5, 10.])
    assert torch.allclose(out, expected)

def test_against_torch():
    start, end, steps = 1.3, 9.7, 17
    out = my_linspace(start, end, steps)
    expected = torch.linspace(start, end, steps)
    assert torch.allclose(out, expected)

def test_single_step():
    out = my_linspace(5, 10, 1)
    expected = torch.tensor([5.])
    assert torch.allclose(out, expected)

def test_same_start_end():
    out = my_linspace(3, 3, 4)
    expected = torch.tensor([3., 3., 3., 3.])
    assert torch.allclose(out, expected)

def test_negative_range():
    out = my_linspace(-5, 5, 5)
    expected = torch.linspace(-5, 5, 5)
    assert torch.allclose(out, expected)
    
def test_large_step():
    out = my_linspace(0,10,1000)
    expected=torch.linspace(0,10,1000)
    assert torch.allclose(out,expected)
