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
from horton.meanfield.utils import get_level_shift


__all__ = ['FracOccOptimizer']


class FracOccOptimizer(object):
    """A class to optimize orbitals with fractional occupations."""

    def __init__(self, scf_solver=PlainSCFSolver(1e-7), threshold=1e-8):
        """Initialize the optimizer.

        Parameters
        ----------

        scf_solver: ``SCFSolver`` instance
            The SCF solver to use in the inner loop.

        threshold : float
            The convergence threshold for the wavefunction.
        """
        self.scf_solver = scf_solver
        self.threshold = threshold

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
            raise TypeError('The number of initial orbital expansions does not match the Hamiltonian.')

        self.ham = ham
        self.overlap = overlap
        self.get_initial_guess(*orbs)
        if active_orbs:
            self.active_orbs = active_orbs
        else:
            self.active_orbs = self.get_active_orbitals(*orbs)

        if log.do_medium:
            log('Start Fractional Occupations optimizer. ndm=%i' % ham.ndm)
            log.hline()
            log('Iter         Error')
            log.hline()

    def get_energy(occs):

        self.occ_model = FixedOccModel()
        return energy

    def get_gradient(occs):
        return gradient

    def solve_scf(*orbs):
        # Solve SCF
        if self.scf_solver.kind = 'orb':
            self.scf_solver(self.ham, self.overlap, self.occ_model, *orbs)
        else:
            dms = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
            for i, orb in enumerate(orbs):
                dms[i] = orb.to_dm()
            self.scf_solver(self.ham, self.overlap, self.occ_model, *dms)
            ham.reset(*dms)
            focks = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
            ham.compute_fock(*focks)
            for i, orb in enumerate(orbs):
                orb.from_fock_and_dm(focks[i], dms[i], overlap)


    def get_initial_guess(self, *orbs):
        """Use ODA SCF to get an initial guess for fractional occupations"""
        focks = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
        dms = [np.zeros(self.overlap.shape) for i in xrange(self.ham.ndm)]
        nelec = 0
        for i,orb in enumerate(orbs):
            dms[i] = orb.to_dm()
            nelec += sum(orb.occupations[:orb.homo_index()])

        oda = ODASCFSolver(threshold=1e-7)
        occ_model = AufbauOccModel(nelec)
        oda(self.ham, self.overlap, occ_model, *dms)
        ham.reset(*dms)
        ham.compute_fock(*focks)
        for i, orb in enumerate(orbs):
            orb.from_fock_and_dm(focks[i], dms[i], overlap)

    def get_active_orbitals(self, *orbs):
        """Finds orbitals with fractional occupations"""
        actives = []
        for i, orb in enumerate(orbs):
            orb = dms[i].from_fock_and_dm(focks[i], dms[i], overlap)
            lumo = orb.lumo_index()
            tmp = np.where(orb.occupations[:lumo] < 1.)[0]
            if tmp:
                actives.append([j for j in tmp])
            else:
                raise ValueError("No active orbitals found. You need to specify the active orbitals.")
        return actives

    def log(self):
        '''Print headers inside this class'''
        def new_scipy():
            log('============================NEW SCIPY=============================')

        def converged():
            log.hline()
            log('  CONVERGED INSIDE CYCLE - SCIPY')
            log.hline()



class FracOccMSOptimizer(FracOccOptimizer):
    """A class to optimize orbitals with fractional occupations 
    using the Multi-secant Hessian approximation."""

    def __init__(self, maxorbs=5, scf_solver=PlainSCFSolver(1e-7), threshold=1e-8, maxiter=128):
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
        FracOccOptimizer.__init__(scf_solver, threshold, maxiter)

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
        if active_orbs:
            self.active_orbs = active_orbs
        else:
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
        while error > self.threshold or 

            if log.do_medium:
                log('%4i  %12.5e' % (counter, error))

        if log.do_medium:
            log.blank()

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

