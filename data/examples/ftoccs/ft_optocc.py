#!/usr/bin/env python
#JSON {"lot": "RHF/cc-pvtz",
#JSON  "scf": "PlainSCFSolver",
#JSON  "linalg": "DenseLinalgFactory",
#JSON  "difficulty": 1,
#JSON  "description": "Basic RHF example with dense matrices, includes export of Hamiltonian"}

from horton import *
import numpy as np


# Hartree-Fock calculation
# ------------------------

# Construct a molecule from scratch
mol = IOData(title='Be')
mol.coordinates = np.array([[0.0, 0.0, 0.0]])
mol.numbers = np.array([4])
nelec = 4

# Create a Gaussian basis set
obasis = get_gobasis(mol.coordinates, mol.numbers, 'cc-pvqz')

# Create a linalg factory
lf = DenseLinalgFactory(obasis.nbasis)

# Compute Gaussian integrals
olp = obasis.compute_overlap(lf)
kin = obasis.compute_kinetic(lf)
na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
er = obasis.compute_electron_repulsion(lf)



# Create alpha orbitals
exp_alpha = lf.create_expansion()

# Initial guess
guess_core_hamiltonian(olp, kin, na, exp_alpha)

# Construct the restricted HF effective Hamiltonian
external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
terms = [
    RTwoIndexTerm(kin, 'kin'),
    RDirectTerm(er, 'hartree'),
    RExchangeTerm(er, 'x_hf'),
    RTwoIndexTerm(na, 'ne'),
]
ham = REffHam(terms, external)

# Decide how to occupy the orbitals (6 electrons)
occ_model = AufbauOccModel(2)

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

t_tmp = 0
temp = 6000
highE = -37.693307

dpars = occupationsOptimizer(orb_energies, mu_trial, temp, nelec, eHF, highE)
mu = dpars["mu"]
temp = dpars["temp"]
occ_list = dpars["newOccupations"]

print "mu", mu
print "temp", temp
print "occ list", occ_list
print temp - t_tmp
while abs(temp - t_tmp) > 1e-8:
    print "entre al ciclo"
    mu_trial = mu
    temp = t_tmp
    new_occs = np.array(occ_list[:obasis.nbasis])
    print "new_occs", new_occs

    # Decide how to occupy the orbitals (6 electrons)
    occ_model = FixedOccModel(new_occs)

    # Converge WFN with plain SCF
    scf_solver = PlainSCFSolver(1e-6)
    scf_solver(ham, lf, olp, occ_model, exp_alpha)
    eHF =  ham.cache['energy']

    orb_energies = np.append(exp_alpha.energies, exp_alpha.energies)
    print "orbital energies", orb_energies
    mu, t_tmp, occ_list = occupationsOptimizer(orb_energies, mu_trial, temp, nelec, eHF, highE)
    mu = dpars["mu"]
    temp = dpars["temp"]
    occ_list = dpars["newOccupations"]
    print "occ list again", occ_list


