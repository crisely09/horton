"""
Tests for BaseApproximant class.
"""
from funapp.base import BaseApproximant

def test_base0():
    """
    Test class attributes.
    """
    x = [0., 1.2, 3.5, 3.5, 3.5, 4.7, 7., 7., 7., 9.0]
    y = [1.3, 1.4, 2.3, 3.4, 3.4, 4.5, 5.4, 5.6, 5.5, 6.7]

    base = BaseApproximant(x, y)

    assert base._xlen == len(x)-1
    assert base._ider == [[3.5, [3]], [7.0, [6, 7]]]

test_base0()

def test_base1():
    """
    Test class attributes.
    """
    x = [0., 1.2, 3.5, 3.5, 3.5, 3.5, 3.5, 4.7, 7., 7., 7., 9.0]
    y = [1.3, 1.4, 2.3, 3.4, 3.4, 3.5, 3.5, 4.5, 5.4, 5.6, 5.5, 6.7]

    base = BaseApproximant(x, y)

    assert base._xlen == len(x)-2
    assert base._ider == [[3.5, [3, 4]], [7.0, [7, 8]]]

test_base1()
