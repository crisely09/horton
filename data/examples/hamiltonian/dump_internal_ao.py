#!/usr/bin/env python

from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import

# Set up molecule, define basis set
# ---------------------------------
# get the XYZ file from HORTON's test data directory
fn_xyz = context.get_fn('test/water.xyz')
mol = IOData.from_file(fn_xyz)
obasis = get_gobasis(mol.coordinates, mol.numbers, 'cc-pvdz')
lf = CholeskyLinalgFactory(obasis.nbasis)

# Construct Hamiltonian
# ---------------------
mol.lf = lf
mol.kin = obasis.compute_kinetic(lf)
mol.na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers, lf)
mol.er = obasis.compute_electron_repulsion(lf)
mol.core_energy = compute_nucnuc(mol.coordinates, mol.pseudo_numbers)

# Write to a HDF5 file
# --------------------
mol.to_file('hamiltonian_ao.h5')


# CODE BELOW IS FOR horton-regression-test.py ONLY. IT IS NOT PART OF THE EXAMPLE.
rt_results = {
    'kin': mol.kin._array.ravel()[::10],
    'na': mol.na._array.ravel()[::10],
    'er': mol.er._array.ravel()[::1000],
    'core_energy': mol.core_energy,
}
# BEGIN AUTOGENERATED CODE. DO NOT CHANGE MANUALLY.
import numpy as np  # pylint: disable=wrong-import-position
rt_previous = {
    'na': np.array([
        -6.0797251408814796, 0.0, -1.8318359795245827, -2.5150668886409582, 0.0,
        -6.061928702514308, -1.8185022809671232, 0.28314544048255308, -2.5612286623684115,
        0.14889784275391382, -5.1821395397642984, 0.0, -1.6065246363266326, 0.0,
        -2.1193055903316758, -20.636604199574929, 0.0, 3.2140498161241444,
        0.22793252967254229, -2.2275870688292225, -12.890881502223545,
        0.20355767445439643, 0.0, -0.10488028427486198, 0.0, -12.778416588080809, 0.0,
        0.0, 0.0, -1.8185022809671232, -5.4140292951278415, -0.93660650303069326, 0.0,
        0.0, 0.0, -8.4711446777306918, 0.0, 0.0, 0.0, 0.0, -8.4345065860561981,
        -0.88778762785900955, -0.069054309343865417, -2.0036653123921448,
        0.20355767445439643, -8.6284201435269861, 0.0, 0.91259495636933452,
        -1.8318359795245827, 0.0, -4.3875277704507676, -2.7053738330031201, 0.0,
        0.28314544048255308, -0.93660650303069326, -5.6021960154514723, 0.0, 0.0
    ]),
    'core_energy': 9.157175036429987,
    'kin': np.array([
        0.96945722711242355, 0.0, 0.036292401476283279, 0.07995516390306795, 0.0,
        1.8174999999999994, 0.27452182030061434, -9.8774829880917554e-18,
        0.4858203080864158, -0.22493046394464195, 1.8174999999999994, 0.0,
        -0.0029726129513667956, 0.0, 0.018376480943548955, 10.361104211507204, 0.0,
        -0.24514942860277961, 0.0, 0.17334668011290896, 4.3248263329243191,
        7.9993407779445172e-17, 0.0, -4.6184215514858166e-17, 0.0, 4.3248263329243191,
        0.0, 0.0, 0.0, 0.27452182030061434, 0.68824999999999992, 0.093096714010858797,
        0.0, 0.0, 0.0, 4.1475000000000017, 0.0, 0.0, 0.0, 0.0, 4.1475000000000026,
        0.12248253843499435, 0.0, 0.57792718386389474, 7.9993407779445172e-17,
        4.1475000000000026, 0.0, -0.17882475033034573, 0.036292401476283279, 0.0, 0.183,
        0.34077676967706871, 0.0, -9.8774829880917554e-18, 0.093096714010858797,
        1.8174999999999994, 0.0, 0.0
    ]),
    'er': np.array([
        0.25033239567492199, 0.0, 0.17630008788599191, 0.0065734978654330093,
        0.0046964073290911342, 0.0, 0.0, -0.0064629559636422531, -0.00480013068001059,
        -0.0031541190114659098, 0.0, 0.0012217259319068652, 0.0054900863818752848, 0.0,
        0.0097651691552526187, 0.036059303874884027, -0.0018853350923295575,
        0.004893926005645614, 0.0001887973357382781, -0.023070325617852581, 0.0,
        -0.0010050226489016034, 0.0014470358095306939, -0.013460309621518398,
        0.0046474602825373129, -1.2728296831808143e-18, 0.0017236887440976012, 0.0,
        -7.9921912988219616e-34, -1.7461866506894068e-17, 7.7601750150906977e-18, 0.0,
        0.0, 0.0042136461780804503, 0.0, 0.0, 0.0020878984661182871, 0.0, 0.0,
        0.0025950344571634963, -1.4568930910296433e-16, 0.0, 0.0, -4.0716182265900302e-19,
        -5.0154303437260489e-17, 6.2838341321554557e-17, 0.0, -1.3727537721719014e-16,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0069006166900889799, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.3063448988570178e-15, 0.0, 0.0, -1.2439540478727482e-05, 0.0,
        -0.00035925084142473894, 0.0, -0.0025363921207456532, 0.0,
        -0.00033388375900350126, 0.0, -1.9819809459629023e-15, 3.1754425090082746e-15,
        0.0, 0.0, 0.0, 0.0, 0.0, -0.00016516547175034939, 0.0, 0.00015065657649686879,
        0.0, -1.4065850717744276e-16, 1.3178775659709947e-05, -2.853272268345086e-05,
        8.7961398247661302e-16, 2.4888133295609197e-15, 3.0912142438732783e-14, 0.0,
        6.8014458991082829e-05, 3.1123149219056314e-05, 0.0, 0.0, 0.00021623253392928589,
        1.175402785878484e-14, -2.8893090848432695e-05, 0.0, 6.6973800397855726e-15,
        3.8695761907981267e-05, 5.7065035438975563e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -4.2949360438903183e-15, 9.7348875172699942e-15, 0.0, -4.2125469239714865e-16,
        0.0, 0.0, 0.0, -2.3319204986567059e-05, 1.8742304377253141e-15, 0.0,
        -1.7833878253376762e-14, 0.0, 5.833839247579708e-15, 0.0, 0.00010655106838700417,
        0.0, 0.0, 0.0, 2.3949465977577278e-05, 0.0, 0.0, 0.0, 1.7911462247495621e-15,
        2.3960171535325531e-13, 0.0, 0.0, 0.0, 0.0, -3.4026416454474452e-06, 0.0,
        2.0894137640346342e-05, -7.0532377288069042e-14, 0.0, 0.0, -3.5315067532559838e-07
    ]),
}
