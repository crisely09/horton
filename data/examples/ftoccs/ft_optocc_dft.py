#!/usr/bin/env python
#JSON {"lot": "RHF/cc-pvtz",
#JSON  "scf": "PlainSCFSolver",
#JSON  "linalg": "DenseLinalgFactory",
#JSON  "difficulty": 1,
#JSON  "description": "Basic RHF example with dense matrices, includes export of Hamiltonian"}

from horton import *
import numpy as np


# DFT calculation
# ------------------------

# Construct a molecule from scratch
mol = IOData(title='N')
mol.coordinates = np.array([[0.0, 0.0, 0.0]])
mol.numbers = np.array([7])
nelec = 7
highE = -54.403718
#highE = -24.532966
#highE = -37.693307
#highE = -14.573013
temp = 6000
t_tmp = 0
t2 = 0

# Create a Gaussian basis set
obasis = get_gobasis(mol.coordinates, mol.numbers, 'cc-pvqz')

# Create a linalg factory
lf = DenseLinalgFactory(obasis.nbasis)

# Compute Gaussian integrals
olp = obasis.compute_overlap(lf)
kin = obasis.compute_kinetic(lf)
na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
er = obasis.compute_electron_repulsion(lf)

if nelec % 2 == 0:
    print "--------RESTRICTED----------"
    # Create alpha orbitals
    exp_alpha = lf.create_expansion()

    # Initial guess
    guess_core_hamiltonian(olp, kin, na, exp_alpha)

    # Construct the restricted HF effective Hamiltonian
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        RTwoIndexTerm(kin, 'kin'),
        RDirectTerm(er, 'hartree'),
        RGridGroup(obasis, grid, [
            RLibXCLDA('x'),
            RLibXCLDA('c_vwn'),
        ]),
        RTwoIndexTerm(na, 'ne'),
    ]
    ham = REffHam(terms, external)

    # Decide how to occupy the orbitals (6 electrons)
    occ_model = AufbauOccModel(nelec/2)

    # Converge WFN with plain SCF
    scf_solver = PlainSCFSolver(1e-6)
    scf_solver(ham, lf, olp, occ_model, exp_alpha)
    eHF =  ham.cache['energy']

    print type(exp_alpha.energies)
    orb_energies = np.append(exp_alpha.energies, exp_alpha.energies)
    print "orbital energies", orb_energies

    homo = exp_alpha.get_homo_energy()
    lumo = exp_alpha.get_lumo_energy()
    mu_trial = -abs(homo-lumo)/2

    dpars = occupationsOptimizer(orb_energies, mu_trial, temp, nelec, eHF, highE)
    mu = dpars["mu"]
    t_tmp = dpars["temp"]
    occ_list = dpars["newOccupations"]

    print "mu", mu
    print "temp", temp
    print "occ list", occ_list
    while abs(t2 - t_tmp) > 1e-8:
        print "entre al ciclo"
        mu_trial = mu
        t2 = t_tmp
        new_occs = np.array(occ_list[:obasis.nbasis])
        print "new_occs", new_occs
        print "sum of occs", sum(new_occs)
        assert abs(sum(new_occs)-(nelec/2.)) < 1e-5
        
        # Decide how to occupy the orbitals (6 electrons)
        occ_model = FixedOccModel(new_occs)
        
        # Converge WFN with plain SCF
        scf_solver = PlainSCFSolver(1e-6)
        scf_solver(ham, lf, olp, occ_model, exp_alpha)
        eHF =  ham.cache['energy']
        
        orb_energies = np.append(exp_alpha.energies, exp_alpha.energies)
        print "orbital energies", orb_energies
        dpars = occupationsOptimizer(orb_energies, mu_trial, temp, nelec, eHF, highE)
        mu = dpars["mu"]
        t_tmp = dpars["temp"]
        occ_list = dpars["newOccupations"]
        print "occ list again", occ_list
        print "!!!!!!!!!!!!!!!!!!!!!Cambio Temp", abs(t2 - t_tmp)

else:
    print "--------UNRESTRICTED----------"
    # Create alpha orbitals
    exp_alpha = lf.create_expansion()
    exp_beta = lf.create_expansion()

    # Initial guess
    guess_core_hamiltonian(olp, kin, na, exp_alpha, exp_beta)

    # Construct the restricted HF effective Hamiltonian
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UGridGroup(obasis, grid, [
            ULibXCLDA('x'),
            ULibXCLDA('c_vwn'),
        ]),
        UTwoIndexTerm(na, 'ne'),
    ]
    ham = UEffHam(terms, external)

    # Decide how to occupy the orbitals 
    betas = nelec // 2
    alphas = betas + (nelec % 2)
    occ_model = AufbauOccModel(alphas, betas)

    # Converge WFN with plain SCF
    scf_solver = PlainSCFSolver(1e-6)
    scf_solver(ham, lf, olp, occ_model, exp_alpha, exp_beta)
    eHF =  ham.cache['energy']

    print type(exp_alpha.energies)
    orb_energies = np.append(exp_alpha.energies, exp_beta.energies)
    print "orbital energies", orb_energies

    homoa = exp_alpha.get_homo_energy()
    lumoa = exp_alpha.get_lumo_energy()
    homob = exp_beta.get_homo_energy()
    lumob = exp_beta.get_lumo_energy()
    homo = max(homoa, homob)
    lumo = min(lumoa, lumob)
    print "homo", homo
    print "lumo", lumo
    mu_trial = -abs(homo-lumo)/2

    dpars = occupationsOptimizer(orb_energies, mu_trial, temp, nelec, eHF, highE)
    mu = dpars["mu"]
    t_tmp = dpars["temp"]
    occ_list = dpars["newOccupations"]

    print "occ list", occ_list
    while abs(t2 - t_tmp) > 1e-8:
        print "entre al ciclo"
        mu_trial = mu
        t2 = t_tmp
        new_alpha_occs = np.array(occ_list[:obasis.nbasis])
        new_beta_occs = np.array(occ_list[obasis.nbasis:])
        print "new_occs", new_alpha_occs
        print "sum of occs", sum(new_alpha_occs + new_beta_occs)
        assert abs(sum(new_alpha_occs + new_beta_occs) - nelec) < 1e-5
        
        # Decide how to occupy the orbitals (6 electrons)
        occ_model = FixedOccModel(new_alpha_occs, new_beta_occs)
        
        # Converge WFN with plain SCF
        scf_solver = PlainSCFSolver(1e-6)
        scf_solver(ham, lf, olp, occ_model, exp_alpha, exp_beta)
        eHF =  ham.cache['energy']
        
        orb_energies = np.append(exp_alpha.energies, exp_beta.energies)
        print "orbital energies", orb_energies
        dpars = occupationsOptimizer(orb_energies, mu_trial, temp, nelec, eHF, highE)
        mu = dpars["mu"]
        t_tmp = dpars["temp"]
        occ_list = dpars["newOccupations"]
        print "occ list again", occ_list
        print "!!!!!!!!!!!!!!!!!!!!!Cambio Temp", abs(t2 - t_tmp)

