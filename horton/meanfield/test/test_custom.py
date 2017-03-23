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


from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import
from customgrid import RMyDiracExchange  # pylint: disable=wildcard-import,unused-wildcard-import
import numpy as np


def test_myexchange_n2_hfs_sto3g():
    fn_fchk = context.get_fn('test/n2_hfs_sto3g.fchk')
    mol = IOData.from_file(fn_fchk)
    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, random_rotate=False, mode='keep')

    er = mol.obasis.compute_electron_repulsion(mol.lf)
    ham1 = REffHam([RGridGroup(mol.obasis, grid, [RDiracExchange()])])
    ham2 = REffHam([RGridGroup(mol.obasis, grid, [RMyDiracExchange()])])
    ham3 = REffHam([RGridGroup(mol.obasis, grid, [RLibXCLDA('x')])])

    dm_alpha = mol.exp_alpha.to_dm()
    ham1.reset(dm_alpha)
    ham2.reset(dm_alpha)
    ham3.reset(dm_alpha)
    energy1 = ham1.compute_energy()
    energy2 = ham2.compute_energy()
    energy3 = ham3.compute_energy()
    assert abs(energy1 - energy2) < 1e-3
    assert abs(energy3 - energy2) < 1e-3

    op1 = mol.lf.create_two_index()
    op2 = mol.lf.create_two_index()
    ham1.compute_fock(op1)
    ham2.compute_fock(op2)
    assert op1.distance_inf(op2) < 1e-3


# CAUTION! For some reason the values at extra large mu are crazy
# also, values at small mus are not so good.

def test_modified_exchange_simple():
    # Results are compared with values obtained with a Mathematica Notebook
    # rhos for rs = 0.5, 0.9 and 1.0

    rho =  np.array([1.90986, 0.327479, 0.238732])
    mu = 1.0
    c = 1.0
    alpha = 2.0
    result1 = np.array([-0.57982591, -0.39814474, -0.369785])
    result0 = modified_exchange_energy(rho, mu, c, alpha, np.zeros(3))
    assert (abs(result0 - result1) < 1e-5 ).all()


def test_modified_exchange_pot_simple():
    # Results are compared with values obtained with a Mathematica Notebook
    # rhos for rs = 0.5, 0.9 and 1.0
    rho =  np.array([1.90986, 0.327479, 0.238732])
    mu = 1.0
    c = 1.0
    alpha = 2.0
    result1 = np.array([-0.691351, -0.489874, -0.457539])
    result0 = modified_exchange_potential(rho, mu, c, alpha, np.zeros(3))
    assert (abs(result0 - result1) < 1e-5 ).all()


def test_modifiedexchange_n2_hfs_sto3g_1():
    fn_fchk = context.get_fn('test/n2_hfs_sto3g.fchk')
    mol = IOData.from_file(fn_fchk)
    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, random_rotate=False, mode='keep')

    ham1 = REffHam([RGridGroup(mol.obasis, grid, [RDiracExchange()])])
    ham2 = REffHam([RGridGroup(mol.obasis, grid, [RModifiedExchange(mu=100.0, c=0.0, alpha=1.0)])])

    dm_alpha = mol.exp_alpha.to_dm()
    ham1.reset(dm_alpha)
    ham2.reset(dm_alpha)
    energy1 = ham1.compute_energy()
    energy2 = ham2.compute_energy()
    assert abs(energy1 - energy2) < 1e-2
    op1 = mol.lf.create_two_index()
    op2 = mol.lf.create_two_index()
    ham1.compute_fock(op1)
    ham2.compute_fock(op2)
    assert op1.distance_inf(op2) < 1e-2


def test_modifiedexchange_n2_hfs_sto3g_2():
    fn_fchk = context.get_fn('test/n2_hfs_sto3g.fchk')
    mol = IOData.from_file(fn_fchk)
    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, random_rotate=False, mode='keep')

    mu = 100.0
    c = 1.5 * mu
    import math as m
    alpha = (27.0 * mu) / (8 * m.sqrt(m.pi))
    ham1 = REffHam([RGridGroup(mol.obasis, grid, [RDiracExchange()])])
    ham2 = REffHam([RGridGroup(mol.obasis, grid, [RModifiedExchange(mu=mu, c=c, alpha=alpha)])])

    dm_alpha = mol.exp_alpha.to_dm()
    ham1.reset(dm_alpha)
    ham2.reset(dm_alpha)
    energy1 = ham1.compute_energy()
    energy2 = ham2.compute_energy()
    assert abs(energy1 - energy2) < 1e-2
    op1 = mol.lf.create_two_index()
    op2 = mol.lf.create_two_index()
    ham1.compute_fock(op1)
    ham2.compute_fock(op2)
    assert op1.distance_inf(op2) < 1e-2


def test_modifiedexchange_n2_hfs_sto3g_3():
    fn_fchk = context.get_fn('test/n2_hfs_sto3g.fchk')
    mol = IOData.from_file(fn_fchk)
    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, random_rotate=False, mode='keep')

    ham1 = REffHam([RGridGroup(mol.obasis, grid, [RDiracExchange()])])
    ham2 = REffHam([RGridGroup(mol.obasis, grid, [RModifiedExchange(mu=10.0, c=0.0, alpha=1.0)])])
    ham3 = REffHam([RGridGroup(mol.obasis, grid, [RShortRangeAExchange(mu=10.0, c=0.0, alpha=1.0)])])

    dm_alpha = mol.exp_alpha.to_dm()
    ham1.reset(dm_alpha)
    ham2.reset(dm_alpha)
    ham3.reset(dm_alpha)
    energy1 = ham1.compute_energy()
    energy2 = ham2.compute_energy()
    energy3 = ham3.compute_energy()
    energydiff = energy1 - energy2
    assert  abs(energydiff - energy3) < 1e-10

    op1 = mol.lf.create_two_index()
    op2 = mol.lf.create_two_index()
    op3 = mol.lf.create_two_index()
    ham1.compute_fock(op1)
    ham2.compute_fock(op2)
    ham3.compute_fock(op3)
    op_diff = op1.copy()
    op_diff.iadd(op2, -1)
    opdiff = op_diff.distance_inf(op3)
    assert opdiff < 1e-10


def test_modifiedexchange_h3_hfs_321g():
    fn_fchk = context.get_fn('test/h3_hfs_321g.fchk')
    mol = IOData.from_file(fn_fchk)
    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, random_rotate=False, mode='keep')

    mu = 100.0
    c = 1.5 * mu
    import math as m
    alpha = (27.0 * mu) / (8 * m.sqrt(m.pi))
    ham1 = UEffHam([UGridGroup(mol.obasis, grid, [UDiracExchange()])])
    ham2 = UEffHam([UGridGroup(mol.obasis, grid, [UModifiedExchange(mu=mu, c=c, alpha=alpha)])])

    dm_alpha = mol.exp_alpha.to_dm()
    dm_beta = mol.exp_beta.to_dm()
    ham1.reset(dm_alpha, dm_beta)
    ham2.reset(dm_alpha, dm_beta)
    energy1 = ham1.compute_energy()
    energy2 = ham2.compute_energy()
    assert abs(energy1 - energy2) < 1e-2

    fock_alpha1 = mol.lf.create_two_index()
    fock_beta1 = mol.lf.create_two_index()
    fock_alpha2 = mol.lf.create_two_index()
    fock_beta2 = mol.lf.create_two_index()
    ham1.compute_fock(fock_alpha1, fock_beta1)
    ham2.compute_fock(fock_alpha2, fock_beta2)
    assert fock_alpha1.distance_inf(fock_alpha2) < 1e-2
    assert fock_beta1.distance_inf(fock_beta2) < 1e-2


def test_modifiedexchange_h3_hfs_321g_2():
    fn_fchk = context.get_fn('test/h3_hfs_321g.fchk')
    mol = IOData.from_file(fn_fchk)
    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, random_rotate=False, mode='keep')

    mu = 10.0
    c = 1.5 * mu
    import math as m
    alpha = (27.0 * mu) / (8 * m.sqrt(m.pi))
    ham1 = UEffHam([UGridGroup(mol.obasis, grid, [UDiracExchange()])])
    ham2 = UEffHam([UGridGroup(mol.obasis, grid, [UModifiedExchange(mu=mu, c=c, alpha=alpha)])])
    ham3 = UEffHam([UGridGroup(mol.obasis, grid, [UShortRangeAExchange(mu=mu, c=c, alpha=alpha)])])

    dm_alpha = mol.exp_alpha.to_dm()
    dm_beta = mol.exp_beta.to_dm()
    ham1.reset(dm_alpha, dm_beta)
    ham2.reset(dm_alpha, dm_beta)
    ham3.reset(dm_alpha, dm_beta)
    energy1 = ham1.compute_energy()
    energy2 = ham2.compute_energy()
    energy3 = ham3.compute_energy()
    energydiff = energy1 - energy2
    assert  abs(energydiff - energy3) < 1e-7

    fock_alpha1 = mol.lf.create_two_index()
    fock_beta1 = mol.lf.create_two_index()
    fock_alpha2 = mol.lf.create_two_index()
    fock_beta2 = mol.lf.create_two_index()
    fock_alpha3 = mol.lf.create_two_index()
    fock_beta3 = mol.lf.create_two_index()
    ham1.compute_fock(fock_alpha1, fock_beta1)
    ham2.compute_fock(fock_alpha2, fock_beta2)
    ham3.compute_fock(fock_alpha3, fock_beta3)
    fock_diff_alpha = fock_alpha1.copy()
    fock_diff_alpha.iadd(fock_alpha2, -1)
    fock_diff_beta = fock_beta1.copy()
    fock_diff_beta.iadd(fock_beta2, -1)
    opdiff_alpha = fock_diff_alpha.distance_inf(fock_alpha3)
    opdiff_beta = fock_diff_beta.distance_inf(fock_beta3)
    assert opdiff_alpha < 1e-10
    assert opdiff_beta <  1e-10
