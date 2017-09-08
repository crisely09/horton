# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
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
"""Basic Self-Consistent Field (SCF) algorithm."""


import numpy as np

from horton.log import log, timer
from horton.exceptions import NoSCFConvergence
from horton.meanfield.convergence import convergence_error_eigen
from horton.meanfield.scf import *
from horton.meanfield.scf_oda import *
from horton.meanfield.occ import FixedOccModel, AufbauOccModel
from horton.meanfield.utils import get_level_shift
from slsqp import *


__all__ = ['FracOccOptimizer']


class FracOccOptimizer(object):
    """A class to optimize orbitals with fractional occupations."""

    def __init__(self, scf_solver=PlainSCFSolver(1e-7), threshold=1e-8, active_orbs=None):
        """Initialize the optimizer.

        Parameters
        ----------

        scf_solver: ``SCFSolver`` instance
            The SCF solver to use in the inner loop.

        threshold : float
            The convergence threshold for the wavefunction.

        active_orb: np.ndarray
            The indices of the active orbitals to be optimized.
        """
        self.scf_solver = scf_solver
        self.threshold = threshold
        if active_orbs.any():
            assert isinstance(active_orbs, np.ndarray)
        self.active_orbs = active_orbs

    def __call__(self, ham, overlap, *orbs):
        """Find a self-consistent set of orbitals.

        Parameters
        ----------

        ham : EffHam
            An effective Hamiltonian.
        overlap : np.ndarray, shape=(nbasis, nbasis)
            The overlap operator.
        orb1, orb2, ... : Orbitals
            The initial orbitals. The number of dms must match ham.ndm.
        """
        # Some type checking
        if self.active_orbs.any():
            if ham.ndm != len(self.active_orbs):
                raise TypeError('The number of active orbitals does not match the Hamiltonian.')
        if ham.ndm != len(orbs):
            raise TypeError('The number of initial orbital expansions does not match the Hamiltonian.')

        self.ham = ham
        self.overlap = overlap
        self.nfn = orbs[0].nfn
        self.nelec = sum([sum(orb.occupations[:]) for orb in orbs])
        if self.ham.ndm == 1:
            self.nelec *= 2
        self.get_initial_guess(*orbs)
        if not self.active_orbs.any():
            self.active_orbs = self.get_active_orbitals(*orbs)

        self.index_first = [int(min(self.active_orbs[i])) for i in xrange(ham.ndm)]

        occs = self.get_active_occs(*orbs)
        assert len(occs) == self.active_orbs.size

        # Define arguments for optimizer
        self.orbitals = [orb.copy() for orb in orbs]
        fn = self.get_energy
        #print self.get_energy(occs)
        jac = self.get_gradient
        error = self.get_error
        x0 = occs
        x = np.asarray(x0).flatten()
        #fn = float(np.asarray(fn(x)))
        bounds = [(0., 1.)] * len(x0)
        constraints = ({'type': 'eq', 'fun': self.get_error},)
        from scipy import optimize as op
        xfin = fmin_slsqp(fn, x0, bounds=bounds, f_eqcons=self.get_error, fprime=jac)
        self.set_new_occs(xfin)
        for i, orb in enumerate(orbs):
            orb.occupations[:] = self.orbitals[i].occupations
        self.log()
        #try:
        #    fmin_slsqp(fn, x0, jac, bounds=bounds, constraints=constraints)
        #except Exception as e:
        #    raise ValueError("Error -- objective function should "
        #                        +"return a scalar but instead returns "
        #                        + str(fn) + " for input " + str(x0))


    def get_active_occs(self, *orbs):
        import itertools
        occs = [list(orb.occupations[self.active_orbs[i]]) for i, orb in enumerate(orbs)]
        if len(occs) > 1:
            optoccs = list(itertools.chain(occs[0], occs[1]))
        elif len(occs) == 1:
            optoccs = occs[0]
        else:
            raise ValueError('Somehow something is wrong with the occupations.')
        return np.array(optoccs)

    def set_new_occs(self, occs, get=False):
        '''Set the new occupations in the local orbitals

        Arguments
        occs: np.ndarray
            The occupations to be optimized.
        get: bool
            Set to True to return the occupations array(s).
        '''
        lalpha = len(self.active_orbs[0])
        if len(occs) > len(self.active_orbs[0]):
            if self.ham.ndm != 2:
                raise ValueError("The list of occupations to optimize don't match the given hamiltonian.")
        new_occupations = []
        for i in xrange(self.ham.ndm):
            new_occs = np.zeros(self.nfn)
            new_occs[:self.index_first[i]] = 1.0
            new_occs[self.index_first[i] + len(occs):] = 0.0
            new_occs[self.active_orbs[i]] = occs[lalpha*i:lalpha+(len(occs)*i)]
            iactives = self.active_orbs[i]
            self.orbitals[i].occupations[iactives] = occs[lalpha*i:lalpha+(len(occs)*i)]
            new_occupations.append(new_occs)
        if get:
            return new_occupations

    def get_energy(self, occs):
        # Set the new occupations
        energy = 0.
        new_occupations = self.set_new_occs(occs, True)
        self.occ_model = FixedOccModel(*new_occupations)
        energy += self.solve_scf(*self.orbitals)
        return energy

    def get_gradient(self, occs):
        gradient = np.zeros(len(occs))
        lalpha = len(self.active_orbs[0])
        for i in xrange(self.ham.ndm):
            iactives = self.active_orbs[i]
            gradient[lalpha*i:lalpha+(len(occs)*i)] = self.orbitals[i].energies[iactives]
        return gradient

    def get_error(self, occs):
        self.set_new_occs(occs)
        current_nelec = sum([sum(orb.occupations[:]) for orb in self.orbitals])
        if self.ham.ndm == 1:
            current_nelec *= 2
        error = abs(self.nelec - current_nelec) 
        return error

    def solve_scf(self, *orbs):
        # Solve SCF
        if self.scf_solver.kind == 'orb':
            self.scf_solver(self.ham, self.overlap, self.occ_model, *orbs)
            dms = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
            for i, orb in enumerate(orbs):
                dms[i] = orb.to_dm()
            self.ham.reset(*dms)
        else:
            dms = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
            for i, orb in enumerate(orbs):
                dms[i] = orb.to_dm()
            self.scf_solver(self.ham, self.overlap, self.occ_model, *dms)
            self.ham.reset(*dms)
            focks = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
            self.ham.compute_fock(*focks)
            for i, orb in enumerate(orbs):
                orb.from_fock_and_dm(focks[i], dms[i], overlap)
        energy = self.ham.compute_energy()
        return energy.copy()


    def get_initial_guess(self, *orbs):
        """Use ODA SCF to get an initial guess for fractional occupations"""
        focks = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
        dms = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
        noccs = []
        for i,orb in enumerate(orbs):
            dms[i] = orb.to_dm()
            if orb.homo_index != None:
                noccs.append(sum(orb.occupations[:orb.homo_index+1]))
            else:
                noccs.append(0.)

        oda = ODASCFSolver(threshold=1e-7)
        occ_model = AufbauOccModel(*noccs)
        #oda(self.ham, self.overlap, occ_model, *dms)
        try:
            oda(self.ham, self.overlap, occ_model, *dms)
        except NoSCFConvergence:
            log('NoSCFConvergence exception')
        self.ham.reset(*dms)
        self.ham.compute_fock(*focks)
        for i, orb in enumerate(orbs):
            orb.from_fock_and_dm(focks[i], dms[i], self.overlap)

    def get_active_orbitals(self, *orbs):
        """Finds orbitals with fractional occupations"""
        actives = []
        for i, orb in enumerate(orbs):
            homo = orb.homo_index
            tmp = np.where(orb.occupations[homo:] > 1e-7)[0]
            if tmp.any():
                actives.append([j for j in tmp])
            else:
                raise ValueError("No active orbitals found. You need to specify the active orbitals.")
        return np.array(actives)

    def log(self):
        '''Print headers inside this class'''
        log.blank
        log('Results from the Occupations Optimization')
        log.hline()




class FracOccMSOptimizer(FracOccOptimizer):
    """A class to optimize orbitals with fractional occupations 
    using the Multi-secant Hessian approximation."""

    def __init__(self, maxorbs=5, scf_solver=PlainSCFSolver(1e-7), threshold=1e-8,
                 maxiter=128, active_orbitals=None):
        """Initialize the optimizer.

        Parameters
        ----------

        maxorbs: integer
            The number of orbitals to be stored and used
            for the Hessian approximation.

        scf_solver: ``SCFSolver`` instance
            The SCF solver to use in the inner loop.

        threshold : float
            The convergence threshold for the wavefunction.
        """
        self.maxorbs = maxorbs
        FracOccOptimizer.__init__(scf_solver, threshold, active_orbitals)

    def __call__(self, ham, overlap, active_orbs=None, *orbs):
        """Find a self-consistent set of orbitals.

        Parameters
        ----------

        ham : EffHam
            An effective Hamiltonian.
        overlap : np.ndarray, shape=(nbasis, nbasis)
            The overlap operator.
        orb1, orb2, ... : Orbitals
            The initial orbitals. The number of dms must match ham.ndm.
        """
        # Some type checking
        if ham.ndm != len(orbs):
            raise TypeError('The number of initial orbitals does not match the Hamiltonian.')

        # Store active orbitals
        self.norbs = ham.ndm
        self.nfn = orbs[0].nfn
        self.get_initial_guess(ham, overlap, occ_model, *orbs)
        self.set_orbitals(*orbs)
        if not self.active_orbs:
            self.active_orbs = self.get_active_orbitals(*orbs)

        if len(orbs) > 1:
            self.solve_unrestricted(ham, overlap, occ_model, *orbs)
        else:
            self.solve_restricted(ham, overlap, occ_model, *orbs)

        if log.do_medium:
            log('Start Fractional Occupations optimization. ndm=%i' % ham.ndm)
            log.hline()
            log('Iter         Error')
            log.hline()
        error = 0.0
        #while error > self.threshold:
#
#            if log.do_medium:
#                log('%4i  %12.5e' % (counter, error))

        counter = 0.
        if log.do_medium:
            log('Start Fractional Occupations optimizer. ndm=%i' % ham.ndm)
            log.hline()
            log('Iter         Error')
            log.hline()
        if log.do_medium:
            log('%5i %20.13f' % (counter, energy0))

        if not converged:
            raise NoSCFConvergence

    def set_orbitals(*orbs):
        """Store orbital information inside the class"""
        self._occs = np.zeros((self.nfn,self.norbs))
        self._energies = np.zeros((self.nfn, self.norbs))
        self._coeffs = np.zeros((self.nfn, self.norbs))
        for i in xrange(self.norbs):
            self._occs[:,i] = orbs[i].occupations
            self._energies[:,i] = orbs[i].energies
            self._coeffs[:,i] = orbs[i].coeffs

    def _get_orb_occs(self):
        return self._occs.view()

    orb_occs = property(_get_orb_occs)

    def _get_orb_energies(self):
        return self._energies.view()

    orb_energies = property(_get_orb_energies)

    def _get_orb_coeffs(self):
        return self._coeffs.view()

    orb_coeffs = property(_get_orb_coeffs)


    def get_current_occs(self):
        return

    def get_gradient(self):
        return

    def get_hartree_integrals(self):
        jindex = None
        for i,term in enumerate(self.ham.terms):
            if term.label == 'hartree':
                jindex = i
        if jindex:
            hartree_terms = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
            self.ham.terms[i].add_fock(self.ham.cache, *hartree_terms)
        else:
            raise ValueError("Hamiltonian doesn't contain a Hartree term.")
        return hartree_terms

    def compute_quadratic_energy(occs):
        '''Compute the approximate energy for k+1 iteration
            E(occs) = E(occs_k) + grad*(occs - occs_k) + 1/2sum(sum(hess_ij*(occs_i - occs_k,i)*(occs_j - occs_k,j)))

            **Arguments**
            occs
                Numpy 1-dimensional array with the occupation numbers that are optimized
        '''
        scf_energy = self.get_scf_energy()
        occs_initial = self.get_current_occs()
        grad = self.get_orb_energies()
        hess = get_approx_hessian()


    def do_trust_step(Ek, Eq, Emin, occs_k, occs_k1, t):
        '''Make a trust-region step to update occupations
        
            **Arguments**
            Ek
                Energy of the kth iteration
            Eq
                Enery from the minimization of the quadratic model (subproblem)
            Emin
                Actual energy using the occupation numbers after the optimization of the quadratic model
            occs_k, occs_k1
                Orbital occupation numbers from the current kth iteration and from the optimization
                of the quadratic model
        '''
        #Get the radio of the actual reduction and the predicted reduction
        kappa = (Ek - Emin)/(Ek - Eq)

        #Control the size of the step
        if kappa > 3/4.:
            t = max(1, t*(4/3.))
        elif kappa < 1/4. :
            t = t*(3/4.)
        #make the step
        occs_k += t*(occs_k1 - occs_k)
        check0 = np.nonzero(occs_k < 0.)
        check1 = np.nonzero(occs_k > 1)
        if check0:
            for i in check0[0]: occs_k[i] = 0.
        if check1:
            for i in check1[0]: occs_k[i] = 1.
        return occs_k, t


def get_inverse(matrix, threshold=1e-6):
    """Compute the inverse of a matrix M using SVD. 
    An approximage inverse matrix is returned when M ill-conditioned or singular

    Arguments
    ---------
    matrix
        The matrix to be inverted.
    threshold
        A threshold for the eigenvalues.
    """
    u, s, vT = numpy.linalg.svd(matrix)
    # check singularity/ill-conditioning
    d = np.zeros(u.shape)
    for i in range(len(s)):
        if s[i] > threshold:
            d[i,i] = 1/s[i]
        else:
            d[i,i] = 0.0
    return np.dot(vT.T, np.dot(d, u.T))


def get_multisecant_hessian(j, u, z):
    '''Construct the approximate Hessian with multi-secant approximation
        H{k} = J{k} - J*U*((U.T*J*U)^-1)*U.T*J + Z*[1/2(U.T*Z + Z.T*U)]^-1*Z.T
        **Arguments**
            j
             Numpy array the Coulomb integrals of the active orbitals
            u
             Numpy array with the n_active Occupations' matrix
            z
             Numpy array with the n_active Gradient matrix
    '''

    Hess = np.copy(j)
    jc=np.copy(j)

    #check if the matrices have the right shape
    assert z.shape == u.shape
    assert j._array.shape[1] == z.shape[0]

    #J*U*((U.T*J*U)^-1)*U.T*J
    uT = np.copy(u.T)
    uTjU = np.dot(uT, np.dot(jc, u))
    uTjui = get_inverse(uTju)
    uTj = np.dot(uT,jc)
    uTjuiuTj2 = np.dot(uTjui,uTj)
    u_term = np.dot(ju, uTjUiuTj2)

    #Z*[1/2(U.T*Z + Z.T*U)]^-1*Z.T
    zT = np.copy(z.T)
    zTu = np.dot(zT, u)
    uTzzTu = 0.5*(zTu + np.dot(uT, z))
    uTzzTui = get_inverse(uTzzTu)
    z_term = np.dot(z, np.dot(uTzzTui, zT))

    assert Hess.shape == z_term.shape and z_term.shape == u_term.shape
    Hess = Hess - u_term + z_term
    return Hess

