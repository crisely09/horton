# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Built-in energy terms"""


import numpy as np

from horton.meanfield.gridgroup import GridObservable, DF_LEVEL_LDA
from horton.grid.molgrid import BeckeMolGrid
from horton.grid.poisson import solve_poisson_becke
from horton.utils import doc_inherit


__all__ = ['RCustomGridObservable', 'RModifiedExchange', 'UModifiedExchange',
            'modified_exchange_energy', 'modified_exchange_potential']


class RCustomGridObservable(GridObservable):
    '''Class used to load potentials in numpy arrays and convert them to
        Hamiltonian terms
    '''
    df_level = DF_LEVEL_LDA

    def __init__(self, pot, label):
        '''
           **Arguments:**

           pot
                A numpy array whit the potential to be used

           label
                A unique label for this contribution
        '''

        self.pot = pot
        GridObservable.__init__(self, label)

    @doc_inherit(GridObservable)
    def compute_energy(self, cache, grid):
        #print list(cache.iterkeys())
        rho = cache['rho_full']
        return grid.integrate(self.pot, rho)

    @doc_inherit(GridObservable)
    def add_pot(self, cache, grid, dpot_alpha):
        dpot_alpha += self.pot


class MyDiracExchange(GridObservable):
    """Common code for the Dirac Exchange Functional implementations."""

    df_level = DF_LEVEL_LDA

    def __init__(self, label='x_asymp', coeff=None):
        r"""Initialize a ModifiedExchange instance.

        Parameters
        ----------
        label : str
            A label for this observable.

        coeff : float
            The coefficient Cx in front of the Dirac exchange energy. It defaults to the
            uniform electron gas value, i.e. :math:`C_x = \frac{3}{4}
            \left(\frac{3}{\pi}\right)^{1/3}`.
        """
        if coeff is None:
            self.coeff = 3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3.0)
        else:
            self.coeff = coeff
        # This part is for the potential evaluation (the derivative of the energy)
        # also, the 2^1/3 comes from the generalization to Restricted Hamiltonians
        # so you do (2 rho)^1/3 or (2 rho)^4/3
        self.derived_coeff = -self.coeff * (4.0 / 3.0) * 2 ** (1.0 / 3.0) 
        GridObservable.__init__(self, label)

    def _update_pot(self, cache, grid, select):
        """Recompute an Exchange potential if invalid.

        Parameters
        ----------
        cache : Cache
            Storage for the potentials.
        grid : IntGrid
            A numerical integration grid.
        select : str
            'alpha' or 'beta'
        """
        rho = cache['all_%s' % select][:, 0]
        pot, new = cache.load('pot_x_dirac_%s' % select, alloc=grid.size)
        if new:
            pot[:] = self.derived_coeff * (rho) ** (1.0 / 3.0)
        return pot


class RMyDiracExchange(MyDiracExchange):
    """The Dirac Exchange Functional for restricted wavefunctions."""

    @doc_inherit(GridObservable)
    def compute_energy(self, cache, grid):
        pot = self._update_pot(cache, grid, 'alpha')
        rho = cache['all_alpha'][:, 0]
        return (3.0 / 2.0) * grid.integrate(pot, rho)

    @doc_inherit(GridObservable)
    def add_pot(self, cache, grid, pots_alpha):
        pots_alpha[:, 0] += self._update_pot(cache, grid, 'alpha')


class ModifiedExchange(GridObservable):
    """Common code for the Dirac Exchange Functional implementations."""

    df_level = DF_LEVEL_LDA

    def __init__(self, label='x_erfgauss', mu=0.0, c=1.0, alpha=1.0):
        r"""Initialize a ModifiedExchange instance.

        Parameters
        ----------
        label : str
            A label for this observable.

        coeff : float
            The coefficient Cx in front of the Dirac exchange energy. It defaults to the
            uniform electron gas value, i.e. :math:`C_x = \frac{3}{4}
            \left(\frac{3}{\pi}\right)^{1/3}`.
        """
        self.mu = mu
        self.c = c
        self.alpha = alpha
        self._coeff = 2.0 ** (1.0/3.0)
        GridObservable.__init__(self, label)

    def _update_pot(self, cache, grid, select):
        """Recompute an Exchange potential if invalid.

        Parameters
        ----------
        cache : Cache
            Storage for the potentials.
        grid : IntGrid
            A numerical integration grid.
        select : str
            'alpha' or 'beta'
        """
        rho = cache['all_%s' % select][:, 0]
        pot, new = cache.load('pot_x_dirac_%s' % select, alloc=grid.size)
        if new:
            pot = self._coeff * modified_exchange_potential(rho, self.mu, self.c, self.alpha, pot)
        return pot

    def _update_ex(self, cache, grid, select):
        """Recompute an Exchange energy per particle if invalid.

        Parameters
        ----------
        cache : Cache
            Storage for the potentials.
        grid : IntGrid
            A numerical integration grid.
        select : str
            'alpha' or 'beta'
        """
        rho = cache['all_%s' % select][:, 0]
        ex, new = cache.load('ex_x_dirac_%s' % select, alloc=grid.size)
        if new:
            ex[:] = self._coeff * modified_exchange_energy(rho, self.mu, self.c, self.alpha, ex)
        return ex

class RModifiedExchange(ModifiedExchange):
    """The Dirac Exchange Functional for restricted wavefunctions."""

    @doc_inherit(GridObservable)
    def compute_energy(self, cache, grid):
        ex = self._update_ex(cache, grid, 'alpha')
        rho = cache['all_alpha'][:, 0]
        return (2.0) * grid.integrate(ex, rho)

    @doc_inherit(GridObservable)
    def add_pot(self, cache, grid, pots_alpha):
        pots_alpha[:, 0] += self._update_pot(cache, grid, 'alpha')


class UModifiedExchange(ModifiedExchange):
    """The Dirac Exchange Functional for unrestricted wavefunctions."""

    @doc_inherit(GridObservable)
    def compute_energy(self, cache, grid):
        ex_alpha = self._update_ex(cache, grid, 'alpha')
        ex_beta = self._update_ex(cache, grid, 'beta')
        rho_alpha = cache['all_alpha'][:, 0]
        rho_beta = cache['all_beta'][:, 0]
        return (grid.integrate(ex_alpha, rho_alpha) +
                              grid.integrate(ex_beta, rho_beta))

    @doc_inherit(GridObservable)
    def add_pot(self, cache, grid, pots_alpha, pots_beta):
        pots_alpha[:, 0] += self._update_pot(cache, grid, 'alpha')
        pots_beta[:, 0] += self._update_pot(cache, grid, 'beta')


def modified_exchange_energy(rho, mu, c, alpha, output):
    '''Compute exchange energy per particle of the asymptotic potential
        V(r) = erf(mu r)/r + c exp(-alpha^2 r^2)
    '''
    import math as m
    from scipy.special import erf as erf

    for i in range(len(rho)):
        if rho[i] < 1e-15:
            output[i] = 0.
        else:
            rho_inv = 1.0 / rho[i]
            var1 = np.power(3.0 * rho[i] * m.pi**2.0, 1.0/3.0)
            var2 = var1 * var1
            mu_sqr = mu * mu
            alpha_sqr = alpha * alpha
            # terms that depend on rho^-1/3
            ex1 = (alpha * c * 3.0 ** (2.0 / 3.0))/(2.0 * m.pi ** (7.0 / 6.0))
            ex1 -= (alpha * c * m.exp(-var2 / (alpha_sqr)))/(2.0 * (3.0 ** (1.0 / 3.0)) * m.pi ** (7.0 / 6.0))
            ex1 += (mu_sqr * 3.0 ** (2.0 / 3.0))/(2 * m.pi ** (5.0 / 3.0))
            ex1 -= (mu_sqr * m.exp(-var2/(mu_sqr)))/((m.pi ** (5.0 / 3.0)) * 3.0 ** (1.0/3.0))
            ex1 *= rho[i] ** (-1.0 / 3.0)
            # terms that depend on rho^-1
            ex2 = -(alpha * alpha_sqr * c) / (3 * m.pi ** (5.0 / 2.0))
            ex2 += (alpha * alpha_sqr * c * m.exp(-var2/alpha_sqr))/(3 * m.pi ** (5.0 / 2.0))
            ex2 -= (mu_sqr * mu_sqr)/ (6.0 * m.pi ** 3.0)
            ex2 += (m.exp(- var2 / mu_sqr) * mu_sqr * mu_sqr) / (6 * m.pi ** 3.0)
            ex2 *= rho_inv
            # terms rho independent
            #ex1 *= rho[i] ** (-1.0 / 3.0)
            ex3 = - 0.5 * c * erf(var1/alpha) - (mu * erf(var1/mu)) / (m.pi ** (1.0 / 2.0))
            output[i] = ex1 + ex2 + ex3
    return output

def modified_exchange_potential(rho, mu, c, alpha, output):
    '''Compute exchange potential of the asymptotic potential
        V(r) = erf(mu r)/r + c exp(-alpha^2 r^2)
    '''
    import math as m
    from scipy.special import erf as erf

    for i in range(len(rho)):
        if rho[i] < 1e-15:
            output[i] += 0.
        else:
            rho_inv = 1.0 / rho[i]
            var1 = np.power(3.0 * rho[i] * m.pi**2.0, 1.0/3.0)
            var2 = var1 * var1
            mu_sqr = mu * mu
            alpha_sqr = alpha * alpha
            # terms that depend on rho^-4/3
            pot1 = (alpha * c)/(m.pi ** (7.0 / 6.0))
            pot1 -= (alpha * c *m.exp(-var2 / (alpha_sqr)))/(m.pi ** (7.0 / 6.0))
            pot1 += (mu_sqr)/(m.pi ** (5.0 / 3.0))
            pot1 -= (mu_sqr * m.exp(-var2/mu_sqr))/(m.pi ** (5.0 / 3.0))
            pot1 *= (3 * rho[i]) ** (-1.0/3.0)
            # terms rho independent
            pot2 = - 0.5 * c * erf(var1/alpha) - (mu * erf(var1/mu)) / (m.pi ** (1.0 / 2.0))
            output[i] += pot1 + pot2
    return output
