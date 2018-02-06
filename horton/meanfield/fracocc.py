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
from horton.meanfield.scf import PlainSCFSolver
from horton.meanfield.scf_oda import ODASCFSolver
from horton.meanfield.occ import FixedOccModel, AufbauOccModel
from horton.meanfield.indextransform import four_index_transform
from slsqp import fmin_slsqp


__all__ = ['FracOccBaseOptimizer', 'FracOccSimpleOptimizer',
           'FracOccMSOptimizer']


class FracOccBaseOptimizer(object):
    """Base class for optimization of orbitals occupations with fractional
       numbers."""

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
            Orbital should be initialized, or
            at least MUST have occupations assigned.
        """
        raise NotImplementedError("Calling base class.")

    def solve_scf(self, ham, overlap, occ_model, *orbs):
        """Solve SCF equations with current orbital occupations
        
        Parameters
        ----------
        ham : ``EffHam`` object
            An effective Hamiltonian.

        overlap : np.ndarray, shape=(nbasis, nbasis)
            The overlap operator.

        occ_model: ``OccModel``
            The choice of occupation model.

        orb1, orb2, ... : Orbitals
            The initial orbitals. The number of dms must match ham.ndm.
            Orbital should be initialized, or at least MUST have occupations 
            assigned.
        """
        # Check type of SCF solver
        if self.scf_solver.kind == 'orb':
            self.scf_solver(ham, overlap, occ_model, *orbs)
            dms = [np.zeros(overlap.shape) for i in xrange(ham.ndm)]
            for i, orb in enumerate(orbs):
                dms[i] = orb.to_dm()
            ham.reset(*dms)
        else:
            dms = [np.zeros(overlap.shape) for i in xrange(ham.ndm)]
            for i, orb in enumerate(orbs):
                dms[i] = orb.to_dm()
            self.scf_solver(ham, overlap, occ_model, *dms)
            ham.reset(*dms)
            focks = [np.zeros(overlap.shape) for i in xrange(ham.ndm)]
            ham.compute_fock(*focks)
            for i, orb in enumerate(orbs):
                orb.from_fock_and_dm(focks[i], dms[i], overlap)
        energy = ham.compute_energy()
        return energy.copy()

    def get_energy(self, occs):
        """Compute the energy of the system with the current occupations.
        Method used only as the function passed to the scipy optimizer

        Parameters
        ----------
        occs: np.ndarray, shape=(sum(len(active_orbs[i])))
            Array with the occupations that are been optimized.
        """
        raise NotImplementedError("Calling base class")

    def get_gradient(self, occs):
        """Compute the gradient of the system with the current occupations.
        Method used only as the function passed to the scipy optimizer

        Parameters
        ----------
        occs: np.ndarray, shape=(sum(len(active_orbs[i])))
            Array with the occupations that are being optimized.
        """
        raise NotImplementedError("Calling base class")

    def get_error(self, occs):
        """Compute the error in the sum of the current occupations and
        the total number of electrons.
        Method used only as the function passed to the scipy optimizer

        Parameters
        ----------
        occs: np.ndarray, shape=(sum(len(active_orbs[i])))
            Array with the occupations that are being optimized.
        """
        raise NotImplementedError("Calling base class")

    def get_initial_guess(self, ham, overlap, *orbs):
        """Use ODA SCF to get an initial guess for fractional occupations

        Parameters
        ----------
        ham : EffHam
            An effective Hamiltonian.

        overlap : np.ndarray, shape=(nbasis, nbasis)
            The overlap operator.

        orb1, orb2, ... : Orbitals
            The initial orbitals. The number of dms must match ham.ndm.
        """
        focks = [np.zeros(overlap.shape) for i in xrange(ham.ndm)]
        dms = [np.zeros(overlap.shape) for i in xrange(ham.ndm)]
        noccs = []
        for i,orb in enumerate(orbs):
            dms[i] = orb.to_dm()
            if orb.homo_index != None:
                noccs.append(sum(orb.occupations[:orb.homo_index+1]))
            else:
                noccs.append(0.)

        oda = ODASCFSolver(threshold=1e-7)
        occ_model = AufbauOccModel(*noccs)
        try:
            oda(ham, overlap, occ_model, *dms)
        except NoSCFConvergence:
            log('NoSCFConvergence exception')
        ham.reset(*dms)
        ham.compute_fock(*focks)
        for i, orb in enumerate(orbs):
            orb.from_fock_and_dm(focks[i], dms[i], overlap)

    def log(self):
        '''Print headers inside this class'''
        log.blank
        log('Results from the Occupations Optimization')
        log.hline()


class FracOccSimpleOptimizer(FracOccBaseOptimizer):
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
        FracOccBaseOptimizer.__init__(self, scf_solver, threshold, active_orbs)

    @doc_inherit(FracOccBaseOptimizer)
    def __call__(self, ham, overlap, *orbs):
        # Some type checking
        if self.active_orbs.any():
            if ham.ndm != len(self.active_orbs):
                raise TypeError('The number of active orbitals does not match the Hamiltonian.')
        if ham.ndm != len(orbs):
            raise TypeError('The number of initial orbitals does not match the Hamiltonian.')

        # Assign local variables
        self.ham = ham
        self.overlap = overlap
        self.nfn = orbs[0].nfn
        self.nelec = sum([sum(orb.occupations[:]) for orb in orbs])
        if self.ham.ndm == 1:
            self.nelec *= 2
        self.get_initial_guess(ham, overlap, *orbs)
        if not self.active_orbs.any():
            self.active_orbs = self.get_active_orbitals(*orbs)

        self.index_first = [int(min(self.active_orbs[i])) for i in xrange(ham.ndm)]

        occs = self.get_active_occs(*orbs)
        assert len(occs) == self.active_orbs.size

        # Define arguments for optimizer
        self.orbitals = [orb.copy() for orb in orbs]
        fn = self.get_energy
        jac = self.get_gradient
        error = self.get_error
        x0 = occs
        x = np.asarray(x0).flatten()
        bounds = [(0., 1.)] * len(x0)
        constraints = ({'type': 'eq', 'fun': self.get_error},)

        # Minimize energy
        xfin = fmin_slsqp(fn, x0, bounds=bounds, f_eqcons=self.get_error, fprime=jac)
        self.update_orbitals(*orbs)
        self.set_new_occs(xfin)
        self.log()

    def update_orbitals(self, *orbs):
        """Store orbital information inside the class"""
        for i, orb in enumerate(orbs):
            orb.occupations[:] = self.orbitals[i].occupations
            orb.energies[:] = self.orbitals[i].energies
            orb.coeffs[:] = self.orbitals[i].coeffs

    def set_new_occs(self, occs, get=False):
        '''Set the new occupations in the local orbitals

        Parameters
        ----------
        occs: np.ndarray
            The occupations to be optimized.

        get: bool
            Set to True to return the occupations array(s).
        '''
        lalpha = len(self.active_orbs[0])
        if len(occs) > len(self.active_orbs[0]):
            if self.ham.ndm != 2:
                raise ValueError("The list of occupations to optimize don't match the Hamiltonian.")
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

    @doc_inherit(FracOccBaseOptimizer)
    def get_energy(self, occs):
        energy = 0.
        # Set the new occupations
        new_occupations = self.set_new_occs(occs, True)
        self.occ_model = FixedOccModel(*new_occupations)
        energy += self.solve_scf(self.ham, self.overlap, self.occ_model, *self.orbitals)
        return energy

    @doc_inherit(FracOccBaseOptimizer)
    def get_gradient(self, occs):
        gradient = np.zeros(len(occs))
        lalpha = len(self.active_orbs[0])
        for i in xrange(self.ham.ndm):
            iactives = self.active_orbs[i]
            gradient[lalpha*i:lalpha+(len(occs)*i)] = self.orbitals[i].energies[iactives]
        return gradient

    @doc_inherit(FracOccBaseOptimizer)
    def get_error(self, occs):
        self.set_new_occs(occs)
        current_nelec = sum([sum(orb.occupations[:]) for orb in self.orbitals])
        if self.ham.ndm == 1:
            current_nelec *= 2
        error = abs(self.nelec - current_nelec) 
        return error

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

    def get_active_occs(self, *orbs):
        """Get a 1-D array with the occupations to optimize

        Parameters
        ----------
        orbs: list, ``Orbital``
            The orbitals being analyzed.
        """
        import itertools
        occs = [list(orb.occupations[self.active_orbs[i]]) for i, orb in enumerate(orbs)]
        if len(occs) > 1:
            optoccs = list(itertools.chain(occs[0], occs[1]))
        elif len(occs) == 1:
            optoccs = occs[0]
        else:
            raise ValueError('Somehow something is wrong with the occupations.')
        return np.array(optoccs)



class FracOccMSOptimizer(FracOccBaseOptimizer):
    """A class to optimize orbitals with fractional occupations 
    using the Multi-secant Hessian approximation."""

    def __init__(self, er, maxorbs=5,
                 scf_solver=PlainSCFSolver(1e-7), threshold=1e-8,
                 maxiter=128, active_orbs=None):
        """Initialize the optimizer.

        Parameters
        ----------
        er: np.ndarray (nbasis, nbasis, nbasis, nbasis,)
            The electron repulsion integrals in the AO basis.

        maxorbs: integer
            The number of orbitals to be stored and used
            for the Hessian approximation.

        scf_solver: ``SCFSolver`` instance
            The SCF solver to use in the inner loop.

        threshold : float
            The convergence threshold for the wavefunction.
        """
        self.er = er
        self.maxorbs = maxorbs
        if not active_orbs.any():
            raise TypeError("For this time of optimizer the active_orbs MUST be provided.")
        self.active_orbs = active_orbs
        ndm = len(active_orbs)
        self.history = FracOccupationHistory(ndm, maxorbs, active_orbs)
        FracOccBaseOptimizer.__init__(self, scf_solver, threshold, active_orbs)

    @doc_inherit(FracOccBaseOptimizer)
    def __call__(self, ham, overlap, *orbs):
        # Some type checking
        if ham.ndm != len(orbs):
            raise TypeError('The number of initial orbitals does not match the Hamiltonian.')

        # Store active orbitals
        self.norbs = ham.ndm
        self.nfn = orbs[0].nfn
        self.get_initial_guess(ham, overlap, *orbs)
        self.orbitals = [orb.copy() for orb in orbs]
        hartree_matrix = self.get_hartree_integrals(*orbs)
 
        error = 0.0
        counter = 0.
        converged = False
        while counter < self.maxiter:

            if log.do_medium:
                log('Start Fractional Occupations optimizer. ndm=%i' % ham.ndm)
                log.hline()
                log('Iter         Error')
                log.hline()

            if self.history.use < self.maxorbs:
                hessian = hartree_matrix
            else:
                occ_matrix, grad_matrix = self.history.get_matrices()
                hessian = get_multisecant_hessian(hartree_matrix, occ_matrix, grad_matrix)

            # Compute quadratic energy
            energy, new_occs = self.minimize_energy(ham, overlap, gradient, hessian, *orbs)
            # Check and sort orbitals
            # Do trust radius
            # Update occupations
            self.history.add(*orbs)
            if log.do_medium:
                log('%4i  %12.5e' % (counter, error))

            if error < self.threshold:
                converged = True
                break

        if not converged:
            raise NoSCFConvergence

    def get_hartree_integrals(self, *orbs):
        """Returns a matrix constructed from the MO Hartree integrals"""
        # Transform integrals to MO basis
        two_mo = []
        for i, orb0 in enumerate(orbs):
            for orb1 in orbs[i:]:
                # Note that integrals are stored using physics' convention <12|12>
                two_mo.append(four_index_transform(self.er, orb0, orb1, orb0, orb1))

        # Check spin polarization and use only the elements
        # corresponding to the active orbitals
        lactives = sum([len(actives) for actives in self.active_orbs])
        hartree_term = np.zeros((lactives, lactives))
        if len(two_mo) > 1:
            assert len(two_mo) == 3
            actives0 = self.active_orbs[0]
            actives1 = self.active_orbs[1]
            lac0 = len(actives0)
            lac1 = len(actives1)
            # Jij = <ij|ij>
            # The hartree term will have alpha-alpha alpha-beta beta-beta blocks
            for k, active0 in enumerate(self.active_orbs):
                for l, active1 in enumerate(self.active_orbs):
                    for i, a in enumerate(active0):
                        for j, b in enumerate(active1):
                            hartree_term[(k*lac0)+i, (l*lac0)+j] = two_mo[k+l][a,b,a,b]
        else:
            if len(two_mo) == 0:
                raise ValueError("No orbitals were provided.")
            # Jij = <ij|ij>
            actives0 = self.active_orbs[0]
            for i, a in enumerate(actives0):
                for j, b in enumerate(actives0):
                    hartree_term[i, j] = two_mo[0][a,b,a,b]
        return hartree_term

    def minimize_energy(self, ham, overlap, gradient, hessian):
        fn = self.get_energy
        jac = self.get_gradient
        error = self.get_error
        x0 = occs
        x = np.asarray(x0).flatten()
        bounds = [(0., 1.)] * len(x0)
        constraints = ({'type': 'eq', 'fun': self.get_error},)

        # Minimize energy
        output = fmin_slsqp(fn, x0, bounds=bounds, f_eqcons=self.get_error, fprime=jac, full_output=True)
        fmin = output[0]
        emin = output[1]

    def get_energy(self, occs):
        r"""Compute the approximate energy for k+1 iteration
        E(occs) = E(occs_k) + grad*(occs - occs_k) 
                + 1/2sum(sum(hess_ij*(occs_i - occs_k,i)*(occs_j - occs_k,j)))

        Parameters
        ----------
        occs:
            Numpy 1-dimensional array with the occupation numbers that are optimized.
        """
        scf_energy = self.solve_scf(self.ham, self.overlap, self.occ_model, *self.orbitals)
        occs_initial = self.get_current_occs()
        grad = self.get_jacobian_matrix()
        hess = get_approx_hessian()
        return energy

    @doc_inherit(FracOccBaseOptimizer)
    def get_gradient(self, occs):
        return gradient

    @doc_inherit(FracOccBaseOptimizer)
    def get_error(self, occs):
        return error

    def do_trust_step(Ek, Eq, Emin, occs_k, occs_k1, t):
        """Make a trust-region step to update occupations
        
        Parameters
        ----------
        Ek
            Energy of the kth iteration
        Eq
            Enery from the minimization of the quadratic model (subproblem)
        Emin
            Actual energy using the occupation numbers after the optimization of the quadratic model
        occs_k, occs_k1
            Orbital occupation numbers from the current kth iteration and from the optimization
            of the quadratic model
        """
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

    Parameters
    ----------
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

    Parameters
    ----------
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


class FracOccStep(object):
    """Class to save information of the orbitals for an individual step"""

    def __init__(self, ndm, nactive_list, *active_orbs):
        """
        Parameters
        ----------
        ndm: integer
            Number of density matrices used to construct the Hamiltonian.

        nactive_list: list, integer
            The number of active orbitals of each spin.

        active_orbs: list, integer
            The indices of the orbitals which occupations are optimized.
        """
        self.active_orbs = active_orbs
        self.nactive_list = nactive_list
        self.total_actives = sum(self.nactive_list)
        self.occupations = np.zeros(self.total_actives)
        self.energies = np.zeros(self.total_actives)

    def clear(self):
        """Clear this record"""
        self.occupations[:] = 0.0
        self.energies[:] = 0.0

    def assign(self, *orbitals):
        """Assing information to the step

        Arguments
        ---------
        orbitals: ``Orbital``
            Current information of the orbitals to be stored.
        """
        for i, orb in enumerate(orbitals):
            init = i * self.nactive_list[i]
            end = init + self.nactive_list[i]
            self.occupations[init:end] = orb.occupations[self.active_orbs]
            self.energies[init:end] = orb.energies[self.active_orbs]


class FracOccupationHistory(object):
    """Class to keep record of fractional occupations' optimization history"""

    def __init__(self, ndm, maxorbs=5, *active_orbs):
        """
        Parameters
        ----------
        ndm: integer
            Number of density matrices used to construct the Hamiltonian.

        maxorbs: integer
            Maximum size of the history.

        active_orbs: list, integer
            The indices of the orbitals which occupations are optimized.
        """
        self.ndm = ndm
        self.maxorbs = maxorbs
        self.active_orbs = active_orbs
        self.nactive_list = [len(actives) for actives in self.active_orbs]
        self.total_actives = sum(self.nactive_list)
        self.stack = [FracOccStep(ndm, self.nactive_list, *active_orbs) for i in xrange(maxorbs)]
        self.used = 0

    def shorten(self):
        """Remove the oldest item from the history"""
        self.used -= 1
        step = self.stack.pop(0)
        step.clear()
        self.stack.append(step)

    def add(self, *orbitals):
        """
        Arguments
        ----------

        orbitals: ``Orbital``
            Current information of the orbitals to be stored.
        """
        if self.ndm != len(orbitals):
            raise TypeError("The number of orbitals must match the ndm parameter")
        # There must be a free spot. If needed, make one.
        if self.used == self.maxorbs:
            self.shorten()
        step = self.stack[self.used]
        step.assign(*orbitals)

        # prepare for next iteration
        self.used += 1

    def get_matrices(self):
        """Return two matrices (occupations and energies) stored"""
        occs_matrix = np.zeros((self.total_actives, self.maxorbs))
        grad_matrix = np.zeros((self.total_actives, self.maxorbs))
        for i in xrange(self.maxorbs):
            step = self.stack[i]
            occs_matrix[:, i] = step.occuptations[:]
            grad_matrix[:, i] = step.energies[:]
        return occs_matrix, grad_matrix

