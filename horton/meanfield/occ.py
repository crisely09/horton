# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2015 The HORTON Development Team
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
#--
"""Occupation number models"""


import numpy as np

from horton.exceptions import ElectronCountError
from horton.quadprog import find_1d_root
from horton.constants import boltzmann
from horton.log import log
from horton.utils import doc_inherit
from horton.meanfield.thetaopt import optimizedOccupations


__all__ = [
    'FixedOccModel', 'AufbauOccModel', 'AufbauSpinOccModel', 'FermiOccModel',
    'MarcosOccModel',
]


class OccModel(object):
    '''Base class for the occupation models'''

    def assign(self, *exps):
        '''Assign occupation numbers to the expansion objects

           **Arguments:**

           exp_alpha, exp_beta, ...
                Expansion objects
        '''
        raise NotImplementedError

    def check_dms(self, overlap, *dms, **kwargs):
        '''Test if the given density matrices contain the right number of electrons

           **Arguments:**

           overlap
                The overlap operator.

           dm1, dm2, ...
                Density matrices to be tested.

           **Optional keyword arguments:**

           eps (default=1e-4)
                The allowed deviation.
        '''
        raise NotImplementedError


class FixedOccModel(OccModel):
    def __init__(self, *occ_arrays):
        self.occ_arrays = occ_arrays

    @doc_inherit(OccModel)
    def assign(self, *exps):
        if len(exps) != len(self.occ_arrays):
            raise TypeError('Expected %i expansion objects, got %i.' % (len(self.nocc), len(exps)))
        for exp, occ_array in zip(exps, self.occ_arrays):
            exp.occupations[:len(occ_array)] = occ_array
            exp.occupations[len(occ_array):] = 0.0

    @doc_inherit(OccModel)
    def check_dms(self, overlap, *dms, **kwargs):
        eps = kwargs.pop('eps', 1e-4)
        if len(kwargs) > 0:
            raise TypeError('Unexpected keyword arguments: %s' % kwargs.keys())
        if len(dms) != len(self.occ_arrays):
            raise TypeError('The number of density matrices is incorrect.')
        for dm, occ_array in zip(dms, self.occ_arrays):
            assert abs(overlap.contract_two('ab,ba', dm) - occ_array.sum()) < eps


class AufbauOccModel(OccModel):
    '''The standard Aufbau occupation number model.

       This model just fills up all the lowest lying orbitals. When the total
       number of electrons in one channel is fractional, the fractional electron
       is put in the HOMO orbital.
    '''

    def __init__(self, *noccs):
        '''
           **Arguments:**

           nalpha, nbeta, ...
                The number of electrons in each channel.
        '''
        for nocc in noccs:
            if nocc < 0:
                raise ElectronCountError('Negative number of electrons is not allowed.')
        if sum(noccs) == 0:
            raise ElectronCountError('At least one electron is required.')
        self.noccs = noccs

    @doc_inherit(OccModel)
    def assign(self, *exps):
        if len(exps) != len(self.noccs):
            raise TypeError('Expected %i expansion objects, got %i.' % (len(self.nocc), len(exps)))
        for exp, nocc in zip(exps, self.noccs):
            if exp.nfn < nocc:
                raise ElectronCountError('The number of orbitals must not be lower than the number of alpha or beta electrons.')
            # It is assumed that the orbitals are sorted from low to high energy.
            if nocc == int(nocc):
                exp.occupations[:nocc] = 1.0
                exp.occupations[nocc:] = 0.0
            else:
                exp.occupations[:int(np.floor(nocc))] = 1.0
                exp.occupations[int(np.floor(nocc))] = nocc - np.floor(nocc)
                exp.occupations[int(np.ceil(nocc)):] = 0.0

    @doc_inherit(OccModel)
    def check_dms(self, overlap, *dms, **kwargs):
        eps = kwargs.pop('eps', 1e-4)
        if len(kwargs) > 0:
            raise TypeError('Unexpected keyword arguments: %s' % kwargs.keys())
        if len(dms) != len(self.noccs):
            raise TypeError('The number of density matrices is incorrect.')
        for dm, nocc in zip(dms, self.noccs):
            assert abs(overlap.contract_two('ab,ba', dm) - nocc) < eps


class AufbauSpinOccModel(OccModel):
    '''This Aufbau model only applies to unrestricted wavefunctions'''
    def __init__(self, nel):
        '''
           **Arguments:**

           nel
                The total number of electrons (alpha + beta)
        '''
        if nel <= 0:
            raise ElectronCountError('The number of electron must be positive.')
        self.nel = nel

    @doc_inherit(OccModel)
    def assign(self, exp_alpha, exp_beta):
        nel = self.nel
        ialpha = 0
        ibeta = 0
        while nel > 0:
            if exp_alpha.energies[ialpha] <= exp_beta.energies[ibeta]:
                exp_alpha.occupations[ialpha] = min(1.0, nel)
                ialpha += 1
            else:
                exp_beta.occupations[ibeta] = min(1.0, nel)
                ibeta += 1
            nel -= 1

    @doc_inherit(OccModel)
    def check_dms(self, overlap, *dms, **kwargs):
        eps = kwargs.pop('eps', 1e-4)
        if len(kwargs) > 0:
            raise TypeError('Unexpected keyword arguments: %s' % kwargs.keys())
        assert abs(sum(overlap.contract_two('ab,ba', dm) for dm in dms) - self.nel) < eps


class FermiOccModel(AufbauOccModel):
    '''Fermi smearing electron occupation model'''
    def __init__(self, *noccs, **kwargs):
        r'''
           **Arguments:**

           nalpha, nbeta, ...
                The number of electrons in each channel.

           **Optional keyword arguments:**

           temperature
                Controls the width of the distribution (derivative)

           eps
                The error on the sum of the occupation number when searching for
                the right Fermi level.

           For each channel, the orbital occupations are assigned with the Fermi
           distribution:

           .. math::

                n_i = \frac{1}{1 + e^{(\epsilon_i - \mu)/k_B T}}

           where, for a given set of energy levels, :math:`\{\epsilon_i\}`, the
           chemical potential, :math:`\mu`, is optimized as to satisfy the
           following constraint:

           .. math::

               \sum_i n_i = n_\text{occ}

           where :math:`n_\text{occ}` can be set per (spin) channel. This is
           only a part of the methodology presented in [rabuck1999]_.
        '''
        temperature = kwargs.pop('temperature', 300)
        eps = kwargs.pop('eps', 1e-8)
        if len(kwargs) > 0:
            raise TypeError('Unknown keyword arguments: %s' % kwargs.keys())
        if temperature <= 0:
            raise ValueError('The temperature must be strictly positive')
        if eps <= 0:
            raise ValueError('The root-finder threshold (eps) must be strictly positive.')
        self.temperature = float(temperature)
        self.eps = eps
        AufbauOccModel.__init__(self, *noccs)
        log.cite('rabuck1999', 'the Fermi broading method to assign orbital occupations')

    @doc_inherit(OccModel)
    def assign(self, *exps):
        beta = 1.0/self.temperature/boltzmann
        for exp, nocc in zip(exps, self.noccs):
            def get_occ(mu):
                occ = np.zeros(exp.nfn)
                mask = exp.energies < mu
                e = np.exp(beta*(exp.energies[mask] - mu))
                occ[mask] = 1.0/(e + 1.0)
                mask = ~mask
                e = np.exp(-beta*(exp.energies[mask] - mu))
                occ[mask] = e/(1.0 + e)
                return occ

            def error(mu):
                return nocc - get_occ(mu).sum()

            mu0 = exp.energies[exp.nfn/2]
            error0 = error(mu0)
            delta = 0.1*(1 - 2*(error0 < 0))
            for i in xrange(100):
                mu1 = mu0 + delta
                error1 = error(mu1)
                if error1 == 0 or ((error0 > 0) ^ (error1 > 0)):
                    break
                delta *= 2

            if error1 == 0:
                exp.occupations[:] = get_occ(mu1)
            else:
                mu, error = find_1d_root(error, (mu0, error0), (mu1, error1), eps=self.eps)
                exp.occupations[:] = get_occ(mu)


class MarcosOccModel(AufbauOccModel):
    '''Occupation model for the optimization of the temperature and chemical potential'''
    def __init__(self, *noccs,  **kwargs):
        '''
           **Arguments:**

           nalpha, nbeta, ...
                The number of electrons in each channel.


           theta
                Controls the width of the distribution (derivative)


           **Optional keyword arguments:**

           eps
                The error on the sum of the occupation number when searching for
                the right Fermi level.

           For each channel, the orbital occupations are assigned with the Fermi
           distribution:

           .. math::

                n_i = \frac{1}{1 + e^{(\epsilon_i - \mu)/k_B T}}

           where, for a given set of energy levels, :math:`\{\epsilon_i\}`, the
           chemical potential, :math:`\mu`, is optimized as to satisfy the
           following constraint:

        '''
        eps = kwargs.pop('eps', 1e-8)
        theta = kwargs.pop('theta', 0.3)
        if len(kwargs) > 0:
            raise TypeError('Unknown keyword arguments: %s' % kwargs.keys())
        if theta <= 0:
            raise ValueError('Theta must be strictly positive')
        if eps <= 0:
            raise ValueError('The root-finder threshold (eps) must be strictly positive.')
        self.theta = float(theta)
        self.eps = eps
        if len(noccs) == 1:
            self.nelec = 2*noccs[0]
        else:
            self.nelec = sum(noccs)
        print 'nelec', self.nelec
        self.mu = None
        AufbauOccModel.__init__(self, *noccs)
    
    def get_occupations(self, energies):
        '''Calls external optimization Module to get the new occupations
            after optimizing Teta and mu'''
        print 'previous theta', self.theta
        print 'previous mu', self.mu
        mu, theta, occs = optimizedOccupations(energies, self.mu, self.nelec, self.theta)
        self.theta = theta
        self.mu =  mu
        print 'new theta', self.theta
        print 'new mu', self.mu
        print "########################## occs ", occs
        return occs

    @doc_inherit(OccModel)
    def assign(self, *exps):
        if not  self.mu :
            AufbauOccModel.assign(self, *exps)
            mu = []
            for exp in exps:
                fgap = -abs(exp.get_lumo_energy() - exp.get_homo_energy())/2.0
                mu.append(fgap)
            self.mu = reduce(lambda x, y: x + y, mu) / len(mu)
            print 'start mu', self.mu
        energies = []
        if len(exps) == 1:
            energies.extend(exps[0].energies[:])
            energies = 2*energies
        else:
            for exp in exps:
                energies.extend(exp.energies[:])
        lexp = len(exps[0].energies)
        le = len(energies)
        print 'energies', energies
        occs_tmp = self.get_occupations(energies)
        i = 1
        start = 0
        for exp in exps:
            exp.occupations[:] = occs_tmp[start:i*lexp]
            print 'occupations to exp', occs_tmp[start:i*lexp]
            start = i*lexp


