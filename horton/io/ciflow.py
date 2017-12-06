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
'''CIFlow format.

   .. note ::

       One- and two-electron integrals are stored in chemists' notation in an
       psi4 output  file while HORTON internally uses Physicist's notation.
'''


__all__ = ['dump_psi4']


def dump_psi4(filename, data):
    '''Write one- and two-electron integrals in the Psi4 output format.

       Works only for restricted wavefunctions.

       filename
            The filename of the ciflow file. This is usually ".dat".

       data
            An IOData instance. Must contain ``one_mo``, ``two_mo``.
            May contain ``core_energy``, ``nelec`` and ``ms``
    '''
    one_mo = data.one_mo
    two_mo = data.two_mo
    nactive = data.obasis.nbasis
    core_energy = getattr(data, 'core_energy', 0.0)
    energy = getattr(data, 'energy', 0.0)
    nelec = getattr(data, 'nelec', 0)
    if hasattr(data, 'exp_alpha'):
        nalpha = sum(data.exp_alpha.occupations)
        if hasattr(data, 'exp_beta'):
            nbeta = sum(data.exp_beta.occupations)
        else:
            nbeta = nalpha
    else:
        nbeta = nelec/2
        nalpha = nbeta + (nelec % 2)

    with open(filename, 'w') as f:

        # Write header
        print >> f, '****  Molecular Integrals For DOCI Start Here'
        print >> f, 'Nalpha = %d ' % nalpha
        print >> f, 'Nbeta = %d ' % nbeta
        print >> f, 'Symmetry Label = c1 '
        print >> f, 'Nirreps = 1 '
        print >> f, 'Nuclear Repulsion Energy = %.18f ' % core_energy
        print >> f, 'Number Of Molecular Orbitals =  %d ' % nactive
        print >> f, 'Irreps Of Molecular Orbitals = '
        print >> f, '0 '*nactive
        print >> f, 'DOCC =  0  0  0  0 #this line is ignored '
        print >> f, 'SOCC =  0  0  0  0 #this line is ignored '


        # Write integrals
        print >> f, '****  MO OEI '
        for i in xrange(nactive):
            for j in xrange(i+1):
                value = one_mo[i,j]
                if abs(value - 0.0) > 1e-10:
                    print >> f, '%d %d %.18f' % (j, i, value)

        print >> f, '****  MO TEI '
        for i in xrange(nactive):
            for j in xrange(i+1):
                for k in xrange(nactive):
                    for l in xrange(k+1):
                        if (i*(i+1))/2+j >= (k*(k+1))/2+l:
                            value = two_mo[i,k,j,l]
                            if abs(value - 0.0) > 1e-10:
                                print >> f, '%d %d %d %d %.18f' % (l, k, j, i, value)


        # Write footer
        print >> f, '****  HF Energy = %.18f' % energy
        print >> f, '****    Molecular Integrals For DOCI End Here'
