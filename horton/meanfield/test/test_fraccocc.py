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


from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import
from horton.meanfield.test.common import check_interpolation, helper_compute
import numpy as np


def test_energy_hydrogen():
    fn_fchk = context.get_fn('test/h_sto3g.fchk')
    mol = IOData.from_file(fn_fchk)
    kin = mol.obasis.compute_kinetic()
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    er = mol.obasis.compute_electron_repulsion()
    olp = mol.obasis.compute_overlap()
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UExchangeTerm(er, 'x_hf'),
        UTwoIndexTerm(na, 'ne'),
    ]
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    ham = UEffHam(terms, external)
    fracocc = FracOccSimpleOptimizer(active_orbs=[0])
    fracocc(ham, olp, mol.orb_alpha, mol.orb_beta)

#test_energy_hydrogen()

def test_carbon_restricted():
    coordinates = np.array([[0., 0., 0.]])
    numbers = np.array([6])
    obasis = get_gobasis(coordinates, numbers, '3-21g')
    mol = IOData(numbers=numbers, coordinates=coordinates, obasis=obasis)
    kin = mol.obasis.compute_kinetic()
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    er = mol.obasis.compute_electron_repulsion()
    olp = mol.obasis.compute_overlap()
    # Create alpha orbitals
    mol.orb_alpha = Orbitals(obasis.nbasis)
    one = kin + na
    guess_core_hamiltonian(olp, one, mol.orb_alpha)
    terms = [
        RTwoIndexTerm(kin, 'kin'),
        RDirectTerm(er, 'hartree'),
        RExchangeTerm(er, 'x_hf'),
        RTwoIndexTerm(na, 'ne'),
    ]
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    ham = REffHam(terms, external)
    # Decide how to occupy the orbitals (5 alpha electrons)
    occ_model = AufbauOccModel(3)

    scf_solver = ODASCFSolver(1e-7)
    occ_model.assign(mol.orb_alpha)
    dm_alpha = mol.orb_alpha.to_dm()
    scf_solver(ham, olp, occ_model, dm_alpha)
    ham.reset(dm_alpha)
    fock_alpha = np.zeros(dm_alpha.shape)
    ham.compute_fock(fock_alpha)
    mol.orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
    print 'alpha', mol.orb_alpha.occupations
    fracocc = FracOccSimpleOptimizer(active_orbs=np.array([[2,3,4]]))
    fracocc(ham, olp, mol.orb_alpha)

test_carbon_restricted()

def test_carbon():
    coordinates = np.array([[0., 0., 0.]])
    numbers = np.array([6])
    obasis = get_gobasis(coordinates, numbers, '3-21g')
    mol = IOData(numbers=numbers, coordinates=coordinates, obasis=obasis)
    kin = mol.obasis.compute_kinetic()
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    er = mol.obasis.compute_electron_repulsion()
    olp = mol.obasis.compute_overlap()
    # Create alpha orbitals
    mol.orb_alpha = Orbitals(obasis.nbasis)
    mol.orb_beta = Orbitals(obasis.nbasis)
    one = kin + na
    guess_core_hamiltonian(olp, one, mol.orb_alpha, mol.orb_beta)
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UExchangeTerm(er, 'x_hf'),
        UTwoIndexTerm(na, 'ne'),
    ]
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    ham = UEffHam(terms, external)
    # Decide how to occupy the orbitals (5 alpha electrons)
    occ_model = AufbauOccModel(3,3)

    scf_solver = ODASCFSolver(1e-7)
    occ_model.assign(mol.orb_alpha, mol.orb_beta)
    dm_alpha = mol.orb_alpha.to_dm()
    dm_beta = mol.orb_beta.to_dm()
    scf_solver(ham, olp, occ_model, dm_alpha, dm_beta)
    #scf_solver(ham, olp, occ_model, mol.orb_alpha, mol.orb_beta)
    ham.reset(dm_alpha, dm_beta)
    fock_alpha = np.zeros(dm_alpha.shape)
    fock_beta = np.zeros(dm_beta.shape)
    ham.compute_fock(fock_alpha, fock_beta)
    mol.orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
    mol.orb_beta.from_fock_and_dm(fock_beta, dm_beta, olp)
    print 'alpha', mol.orb_alpha.occupations
    print 'beta', mol.orb_beta.occupations
    fracocc = FracOccSimpleOptimizer(active_orbs=np.array([[1,2,3],[1,2,3]]))
    fracocc(ham, olp, mol.orb_alpha, mol.orb_beta)
    print 'final occs ', mol.orb_alpha.occupations

#test_carbon()


def test_carbon_ms_restricted():
    coordinates = np.array([[0., 0., 0.]])
    numbers = np.array([6])
    obasis = get_gobasis(coordinates, numbers, '3-21g')
    mol = IOData(numbers=numbers, coordinates=coordinates, obasis=obasis)
    kin = mol.obasis.compute_kinetic()
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    er = mol.obasis.compute_electron_repulsion()
    olp = mol.obasis.compute_overlap()
    # Create alpha orbitals
    mol.orb_alpha = Orbitals(obasis.nbasis)
    one = kin + na
    guess_core_hamiltonian(olp, one, mol.orb_alpha)
    terms = [
        RTwoIndexTerm(kin, 'kin'),
        RDirectTerm(er, 'hartree'),
        RExchangeTerm(er, 'x_hf'),
        RTwoIndexTerm(na, 'ne'),
    ]
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    ham = REffHam(terms, external)
    # Decide how to occupy the orbitals (5 alpha electrons)
    occ_model = AufbauOccModel(3)

    scf_solver = ODASCFSolver(1e-7)
    occ_model.assign(mol.orb_alpha)
    dm_alpha = mol.orb_alpha.to_dm()
    scf_solver(ham, olp, occ_model, dm_alpha)
    ham.reset(dm_alpha)
    fock_alpha = np.zeros(dm_alpha.shape)
    ham.compute_fock(fock_alpha)
    mol.orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
    fracocc = FracOccMSOptimizer(er, active_orbs=np.array([[2,3,4]]))
    hartree_matrix = fracocc.get_hartree_integrals(mol.orb_alpha)
    print "hartree matrix ", hartree_matrix
    jm = four_index_transform(er, mol.orb_alpha)

    actives = [2, 3, 4]
    hartree_terms = np.array([jm[i, j, i, j] for i in actives for j in actives])
    hartree_matrix = np.reshape(np.stack(hartree_matrix), (9,))
    assert abs(hartree_terms - hartree_matrix).all() < 1e-6

#test_carbon_ms_restricted()

def test_carbon_ms_hartree_unrestricted1():
    coordinates = np.array([[0., 0., 0.]])
    numbers = np.array([6])
    obasis = get_gobasis(coordinates, numbers, '3-21g')
    mol = IOData(numbers=numbers, coordinates=coordinates, obasis=obasis)
    kin = mol.obasis.compute_kinetic()
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    er = mol.obasis.compute_electron_repulsion()
    olp = mol.obasis.compute_overlap()
    # Create alpha orbitals
    mol.orb_alpha = Orbitals(obasis.nbasis)
    mol.orb_beta = Orbitals(obasis.nbasis)
    one = kin + na
    guess_core_hamiltonian(olp, one, mol.orb_alpha, mol.orb_beta)
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UExchangeTerm(er, 'x_hf'),
        UTwoIndexTerm(na, 'ne'),
    ]
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    ham = UEffHam(terms, external)
    # Decide how to occupy the orbitals (5 alpha electrons)
    occ_model = AufbauOccModel(3,3)

    scf_solver = ODASCFSolver(1e-7)
    occ_model.assign(mol.orb_alpha, mol.orb_beta)
    dm_alpha = mol.orb_alpha.to_dm()
    dm_beta = mol.orb_beta.to_dm()
    scf_solver(ham, olp, occ_model, dm_alpha, dm_beta)
    #scf_solver(ham, olp, occ_model, mol.orb_alpha, mol.orb_beta)
    ham.reset(dm_alpha, dm_beta)
    fock_alpha = np.zeros(dm_alpha.shape)
    fock_beta = np.zeros(dm_beta.shape)
    ham.compute_fock(fock_alpha, fock_beta)
    mol.orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
    mol.orb_beta.from_fock_and_dm(fock_beta, dm_beta, olp)
    fracocc = FracOccMSOptimizer(er, active_orbs=np.array([[2,3,4], [2,3,4]]))
    hartree_matrix = fracocc.get_hartree_integrals(mol.orb_alpha, mol.orb_beta)
    print "hartree matrix ", hartree_matrix
    jm = four_index_transform(er, mol.orb_alpha, mol.orb_beta)

    actives = [2, 3, 4]
    hartree_terms = np.array([jm[i, j, i, j] for i in actives for j in actives] * 4)
    hartree_matrix = np.reshape(np.stack(hartree_matrix), (36,))
    assert (hartree_terms - hartree_matrix).all() < 1e-6

#test_carbon_ms_hartree_unrestricted1()

def test_carbon_ms_hartree_unrestricted2():
    coordinates = np.array([[0., 0., 0.]])
    numbers = np.array([6])
    obasis = get_gobasis(coordinates, numbers, '3-21g')
    mol = IOData(numbers=numbers, coordinates=coordinates, obasis=obasis)
    kin = mol.obasis.compute_kinetic()
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    er = mol.obasis.compute_electron_repulsion()
    olp = mol.obasis.compute_overlap()
    # Create alpha orbitals
    mol.orb_alpha = Orbitals(obasis.nbasis)
    mol.orb_beta = Orbitals(obasis.nbasis)
    one = kin + na
    guess_core_hamiltonian(olp, one, mol.orb_alpha, mol.orb_beta)
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UExchangeTerm(er, 'x_hf'),
        UTwoIndexTerm(na, 'ne'),
    ]
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    ham = UEffHam(terms, external)
    # Decide how to occupy the orbitals (5 alpha electrons)
    occ_model = AufbauOccModel(4,2)

    scf_solver = ODASCFSolver(1e-7)
    occ_model.assign(mol.orb_alpha, mol.orb_beta)
    dm_alpha = mol.orb_alpha.to_dm()
    dm_beta = mol.orb_beta.to_dm()
    scf_solver(ham, olp, occ_model, dm_alpha, dm_beta)
    #scf_solver(ham, olp, occ_model, mol.orb_alpha, mol.orb_beta)
    ham.reset(dm_alpha, dm_beta)
    fock_alpha = np.zeros(dm_alpha.shape)
    fock_beta = np.zeros(dm_beta.shape)
    ham.compute_fock(fock_alpha, fock_beta)
    mol.orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
    mol.orb_beta.from_fock_and_dm(fock_beta, dm_beta, olp)
    fracocc = FracOccMSOptimizer(er, active_orbs=np.array([[1, 2, 3], [0, 1]]))
    hartree_integrals = []
    orbs = [mol.orb_alpha, mol.orb_beta]
    for i, orb in enumerate(orbs):
        for orb1 in orbs[i:]:
            hartree_integrals.append(four_index_transform(er, orb, orb1, orb, orb1))

    hartree_matrix = fracocc.get_hartree_integrals(mol.orb_alpha, mol.orb_beta)
    actives_alpha = [1, 2, 3]
    actives_beta = [0, 1]
    total = len(actives_alpha) + len(actives_beta)
    result0 = np.zeros((total, total))
    hartree_terms = []
    jm0 = hartree_integrals[0]
    jm1 = hartree_integrals[1]
    jm2 = hartree_integrals[2]
    # alphas
    hartree_terms.append([jm0[i, j, i, j] for i in actives_alpha for j in actives_alpha])
    for i, a in enumerate(actives_alpha):
        for j, b in enumerate(actives_alpha):
            result0[i, j] = jm0[a, b, a, b]
    # alpha- beta
    hartree_terms.append([jm1[i, j, i, j] for i in actives_alpha for j in actives_beta])
    for i, a in enumerate(actives_alpha):
        for j, b in enumerate(actives_beta):
            result0[i, len(actives_alpha)+j] = jm1[a, b, a, b]
    # beta - alpha
    hartree_terms.append([jm1[i, j, i, j] for i in actives_beta for j in actives_alpha])
    for i, a in enumerate(actives_beta):
        for j, b in enumerate(actives_alpha):
            result0[len(actives_alpha)+i, j] = jm1[a, b, a, b]
    # betas
    hartree_terms.append([jm2[i, j, i, j] for i in actives_beta for j in actives_beta])
    for i, a in enumerate(actives_beta):
        for j, b in enumerate(actives_beta):
            result0[len(actives_alpha)+i, len(actives_alpha)+j] = jm2[a, b, a, b]

    assert (result0 - hartree_matrix).any() < 1e-6

test_carbon_ms_hartree_unrestricted2()

def test_cubic_interpolation_hfs_cs():
    fn_fchk = context.get_fn('test/water_hfs_321g.fchk')
    mol = IOData.from_file(fn_fchk)

    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, random_rotate=False)
    olp = mol.obasis.compute_overlap()
    kin = mol.obasis.compute_kinetic()
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    er = mol.obasis.compute_electron_repulsion()
    terms = [
        RTwoIndexTerm(kin, 'kin'),
        RDirectTerm(er, 'hartree'),
        RGridGroup(mol.obasis, grid, [
            RDiracExchange(),
        ]),
        RTwoIndexTerm(na, 'ne'),
    ]
    ham = REffHam(terms)

    check_interpolation(ham, olp, kin, na, [mol.orb_alpha])


def test_perturbation():
    fn_fchk = context.get_fn('test/n2_hfs_sto3g.fchk')
    mol = IOData.from_file(fn_fchk)
    scf_solver = PlainSCFSolver(maxiter=1024)

    # Without perturbation
    olp = mol.obasis.compute_overlap()
    kin = mol.obasis.compute_kinetic()
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    er = mol.obasis.compute_electron_repulsion()
    terms = [
        RTwoIndexTerm(kin, 'kin'),
        RDirectTerm(er, 'hartree'),
        RExchangeTerm(er, 'x_hf'),
        RTwoIndexTerm(na, 'ne'),
    ]
    ham = REffHam(terms)
    occ_model = AufbauOccModel(7)

    assert convergence_error_eigen(ham, olp, mol.orb_alpha) > 1e-8
    scf_solver(ham, olp, occ_model, mol.orb_alpha)
    assert convergence_error_eigen(ham, olp, mol.orb_alpha) < 1e-8
    energy0 = ham.compute_energy()

    # Construct a perturbation based on the Mulliken AIM operator
    assert mol.obasis.nbasis % 2 == 0
    nfirst = mol.obasis.nbasis / 2
    operator = mol.obasis.compute_overlap().copy()
    operator[:nfirst,nfirst:] *= 0.5
    operator[nfirst:,:nfirst] *= 0.5
    operator[nfirst:,nfirst:] = 0.0

    # Apply the perturbation with oposite signs and check that, because of
    # symmetry, the energy of the perturbed wavefunction is the same in both
    # cases, and higher than the unperturbed.
    energy1_old = None
    for scale in 0.1, -0.1:
        # Perturbation
        tmp = scale*operator
        perturbation = RTwoIndexTerm(tmp, 'pert')
        # Hamiltonian
        terms = [
            RTwoIndexTerm(kin, 'kin'),
            RDirectTerm(er, 'hartree'),
            RExchangeTerm(er, 'x_hf'),
            RTwoIndexTerm(na, 'ne'),
            perturbation,
        ]
        ham = REffHam(terms)
        assert convergence_error_eigen(ham, olp, mol.orb_alpha) > 1e-8
        scf_solver(ham, olp, occ_model, mol.orb_alpha)
        assert convergence_error_eigen(ham, olp, mol.orb_alpha) < 1e-8
        energy1 = ham.compute_energy()
        energy1 -= ham.cache['energy_pert']

        assert energy1 > energy0
        if energy1_old is None:
            energy1_old = energy1
        else:
            assert abs(energy1 - energy1_old) < 1e-7


def test_ghost_hf():
    fn_fchk = context.get_fn('test/water_dimer_ghost.fchk')
    mol = IOData.from_file(fn_fchk)
    olp = mol.obasis.compute_overlap()
    kin = mol.obasis.compute_kinetic()
    na = mol.obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, )
    er = mol.obasis.compute_electron_repulsion()
    terms = [
        RTwoIndexTerm(kin, 'kin'),
        RDirectTerm(er, 'hartree'),
        RExchangeTerm(er, 'x_hf'),
        RTwoIndexTerm(na, 'ne'),
    ]
    ham = REffHam(terms)
    # The convergence should be reasonable, not perfect because of limited
    # precision in Gaussian fchk file:
    assert convergence_error_eigen(ham, olp, mol.orb_alpha) < 1e-5
