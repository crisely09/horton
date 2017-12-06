"""
Family of Pade Approximants.
"""

import numpy as np
import scipy as sp
from scipy import linalg as sl


__all__ = ['GeneralPadeApproximant',]


class GeneralPadeApproximant(BaseApproximant):
    """
    Class for Pade Approximants generated without Taylor Series
    coefficients.


    Parameters
    ----------
    x: np.ndarray
        Points where the function was evaluated.
    y: np.ndarray
        Values of the function, and/or function
        derivatives evaluated at `x`.
    m, n: int
        Order of numerator and denominator series
        of the Padé approximant.

    Methods
    -------
    __call__
    _get_matrix_form
    _evaluate
    _derivate


    Example
    -------
    >>> x = [0., 1.5, 3.4]
    >>> y = [-0.47, -0.48, -0.498]
    >>> m = 3
    >>> n = 3
    >>> padegen = GeneralPadeApproximant(x, y, m, n)

    Then we evaluate the function and its first derivative on a new set
    of points xnew
    
    >>> xnew = [2.0, 5.0]
    >>> newdata = padegen(xnew, der=[0,1])
    """

    def __init__(self, x, y, m, n):
        # Check arrays and initialize base class
        BaseApproximant.__init__(x, y)
        if m + n <= self._xlen:
            raise ValueError("To construct a [%d/%d] approximant at least\
                    %d points are needed" % (m, n, m+n))
        matrix = self._get_matrix_form()

        # Solve for the coefficients

        # Build approximant using poly1d objects

    def __call__(self, x, der=0):
        """
        Evaluate the Padé approximant and its derivatives
        on the points `x`

        Parameters
        ----------
        x: array like
            Array with points where to evaluate the approximant
        der: int or list(int)
            Number, or list of numbers of the derivative(s) to extract,
            evaluation of the function corresponds to the 0th derivative.
        """
        # 

    def _get_matrix_form():
        """
        Construct the matrix to use for solving the linear equations.
        """
    def _evaluate(self, x):
        """
        Evaluate the Pade Approximant at the set of points `x`.
        """

    def _derivate(self, x):
        """
        Evaluate the Pade Approximant derivatices at `x`.
        """
