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

from numpy import array, allclose
from nose.plugins.attrib import attr

from horton import context


@attr('regression_check')
def test_hf_dft_rks_water_hybgga():
    ref_result_dm_alpha = array([[  9.65392214e-02,   4.58337521e-02,  -1.85096522e-02,
          4.05449988e-02,   1.24295973e-01,  -9.41095093e-02,
          2.77916454e-07,   1.90995830e-02,   6.34664592e-02,
         -6.24651331e-02,  -7.48563769e-09,  -4.94396444e-03,
         -3.09319601e-07,  -1.17066791e-07,  -3.49501304e-03,
         -9.52604367e-03,  -1.97149973e-02,  -1.88689930e-02],
       [  4.58337521e-02,   2.74142989e-02,   1.05232449e-02,
         -1.67254236e-02,   6.91776398e-02,  -5.44211189e-02,
         -2.72735740e-06,  -3.20534410e-02,   3.53227720e-02,
         -3.81359204e-02,  -2.28055109e-06,  -1.65042647e-03,
         -2.08213709e-07,   5.52102556e-08,  -2.92384258e-03,
         -5.30182620e-03,  -1.88683125e-02,  -8.59662008e-03],
       [ -1.85096522e-02,   1.05232449e-02,   1.04154609e+00,
         -8.54282324e-02,  -5.42565087e-07,  -2.02294820e-02,
         -9.46741805e-08,  -1.39030604e-01,   1.03022041e-07,
         -1.60238621e-02,  -7.44873114e-08,   2.87497855e-03,
         -7.55570298e-08,  -7.65968130e-08,  -3.53298228e-03,
         -1.11776702e-07,  -1.85094713e-02,   1.05225005e-02],
       [  4.05449988e-02,  -1.67254236e-02,  -8.54282324e-02,
          2.49966938e-01,   2.88721091e-06,   4.28385564e-02,
          1.57503663e-06,   2.77253649e-01,   7.95972716e-07,
          4.19932558e-02,   1.15340412e-06,  -5.77167601e-03,
          1.87327459e-07,   1.17734354e-07,   7.59111804e-03,
          2.79221621e-08,   4.05430626e-02,  -1.67245268e-02],
       [  1.24295973e-01,   6.91776398e-02,  -5.42565087e-07,
          2.88721091e-06,   2.65789955e-01,   1.17791750e-06,
          2.14533141e-06,  -2.11099107e-06,   1.35716451e-01,
          5.79408476e-07,   4.60471865e-07,  -4.28199148e-07,
         -1.02508539e-06,  -4.85121350e-07,   6.44252146e-07,
         -2.03707806e-02,  -1.24297546e-01,  -6.91745315e-02],
       [ -9.41095093e-02,  -5.44211189e-02,  -2.02294820e-02,
          4.28385564e-02,   1.17791750e-06,   3.28539539e-01,
          1.64937048e-06,   1.30597320e-01,   3.25304982e-06,
          2.27337873e-01,   1.30778551e-07,   1.17258420e-02,
         -4.11438083e-07,  -1.74957712e-07,   1.63372146e-02,
         -8.65419452e-07,  -9.41097239e-02,  -5.44114496e-02],
       [  2.77916454e-07,  -2.72735740e-06,  -9.46741805e-08,
          1.57503663e-06,   2.14533141e-06,   1.64937048e-06,
          4.13898390e-01,  -1.46369299e-06,   1.41163458e-06,
         -3.43805714e-07,   3.26449849e-01,   2.54528492e-07,
         -7.41834931e-07,  -2.21136393e-02,  -7.08210289e-07,
         -1.01050255e-06,   7.40231183e-07,   1.93447031e-06],
       [  1.90995830e-02,  -3.20534410e-02,  -1.39030604e-01,
          2.77253649e-01,  -2.11099107e-06,   1.30597320e-01,
         -1.46369299e-06,   3.30821784e-01,  -1.12260366e-06,
          1.03428637e-01,  -1.54957645e-06,  -3.16035720e-03,
          9.38901318e-08,   2.71346738e-07,   1.23377673e-02,
          2.30146390e-07,   1.91026028e-02,  -3.20471257e-02],
       [  6.34664592e-02,   3.53227720e-02,   1.03022041e-07,
          7.95972716e-07,   1.35716451e-01,   3.25304982e-06,
          1.41163458e-06,  -1.12260366e-06,   6.92989139e-02,
          2.07932610e-06,   4.84503774e-07,  -9.31979782e-08,
         -5.23429317e-07,  -2.64605629e-07,   4.37805330e-07,
         -1.04016349e-02,  -6.34692270e-02,  -3.53219810e-02],
       [ -6.24651331e-02,  -3.81359204e-02,  -1.60238621e-02,
          4.19932558e-02,   5.79408476e-07,   2.27337873e-01,
         -3.43805714e-07,   1.03428637e-01,   2.07932610e-06,
          1.57938626e-01,  -1.07772888e-06,   7.74731180e-03,
         -2.72559244e-07,  -3.09744260e-08,   1.15793190e-02,
         -5.63248134e-07,  -6.24650285e-02,  -3.81290535e-02],
       [ -7.48563769e-09,  -2.28055109e-06,  -7.44873114e-08,
          1.15340412e-06,   4.60471865e-07,   1.30778551e-07,
          3.26449849e-01,  -1.54957645e-06,   4.84503774e-07,
         -1.07772888e-06,   2.57477454e-01,   1.57032007e-07,
         -5.85093607e-07,  -1.74414648e-02,  -6.15310474e-07,
         -7.02608219e-07,   1.50905832e-06,   2.03737209e-06],
       [ -4.94396444e-03,  -1.65042647e-03,   2.87497855e-03,
         -5.77167601e-03,  -4.28199148e-07,   1.17258420e-02,
          2.54528492e-07,  -3.16035720e-03,  -9.31979782e-08,
          7.74731180e-03,   1.57032007e-07,   6.37850178e-04,
         -2.19037241e-08,  -2.30833026e-08,   4.19198637e-04,
         -5.40329755e-09,  -4.94355171e-03,  -1.64986826e-03],
       [ -3.09319601e-07,  -2.08213709e-07,  -7.55570298e-08,
          1.87327459e-07,  -1.02508539e-06,  -4.11438083e-07,
         -7.41834931e-07,   9.38901318e-08,  -5.23429317e-07,
         -2.72559244e-07,  -5.85093607e-07,  -2.19037241e-08,
          6.03631145e-12,   3.96362828e-08,  -1.50623833e-08,
          7.85678409e-08,   6.49438544e-07,   3.25353831e-07],
       [ -1.17066791e-07,   5.52102556e-08,  -7.65968130e-08,
          1.17734354e-07,  -4.85121350e-07,  -1.74957712e-07,
         -2.21136393e-02,   2.71346738e-07,  -2.64605629e-07,
         -3.09744260e-08,  -1.74414648e-02,  -2.30833026e-08,
          3.96362828e-08,   1.18148090e-03,   3.82934322e-08,
          8.23854436e-08,   2.04763169e-07,  -1.00453595e-09],
       [ -3.49501304e-03,  -2.92384258e-03,  -3.53298228e-03,
          7.59111804e-03,   6.44252146e-07,   1.63372146e-02,
         -7.08210289e-07,   1.23377673e-02,   4.37805330e-07,
          1.15793190e-02,  -6.15310474e-07,   4.19198637e-04,
         -1.50623833e-08,   3.82934322e-08,   9.34878023e-04,
         -8.00382936e-08,  -3.49555698e-03,  -2.92364295e-03],
       [ -9.52604367e-03,  -5.30182620e-03,  -1.11776702e-07,
          2.79221621e-08,  -2.03707806e-02,  -8.65419452e-07,
         -1.01050255e-06,   2.30146390e-07,  -1.04016349e-02,
         -5.63248134e-07,  -7.02608219e-07,  -5.40329755e-09,
          7.85678409e-08,   8.23854436e-08,  -8.00382936e-08,
          1.56126556e-03,   9.52676080e-03,   5.30181651e-03],
       [ -1.97149973e-02,  -1.88683125e-02,  -1.85094713e-02,
          4.05430626e-02,  -1.24297546e-01,  -9.41097239e-02,
          7.40231183e-07,   1.91026028e-02,  -6.34692270e-02,
         -6.24650285e-02,   1.50905832e-06,  -4.94355171e-03,
          6.49438544e-07,   2.04763169e-07,  -3.49555698e-03,
          9.52676080e-03,   9.65404660e-02,   4.58298191e-02],
       [ -1.88689930e-02,  -8.59662008e-03,   1.05225005e-02,
         -1.67245268e-02,  -6.91745315e-02,  -5.44114496e-02,
          1.93447031e-06,  -3.20471257e-02,  -3.53219810e-02,
         -3.81290535e-02,   2.03737209e-06,  -1.64986826e-03,
          3.25353831e-07,  -1.00453595e-09,  -2.92364295e-03,
          5.30181651e-03,   4.58298191e-02,   2.74091910e-02]])
    ref_result_energy = -76.406158025111637
    ref_result_exp_alpha = array([[  1.90403150e-04,  -1.39560167e-01,  -2.41094163e-01,
          1.37607229e-01,   2.03565315e-06,   1.01229117e-01,
          9.56742550e-02,   8.47929942e-01,   8.13482001e-01,
         -4.01430926e-05,  -3.56395292e-01,  -1.22470815e-03,
         -1.74413968e-01,   4.44948921e-06,   9.07853503e-02,
          1.68982540e-05,   8.28626106e-01,  -9.56007786e-01],
       [  3.04376422e-03,  -1.01526698e-03,  -1.34182801e-01,
          9.69483348e-02,  -3.00405722e-06,   9.50645487e-01,
          1.28947926e+00,  -6.04638030e-01,  -5.60979633e-01,
          2.16614261e-05,  -1.50519095e-02,  -9.35459331e-01,
         -6.68559574e-01,   5.54546770e-06,   4.64498045e-02,
         -5.39269981e-06,   1.10749567e-01,   1.16357178e-02],
       [  9.95040947e-01,   2.12402389e-01,  -2.72158565e-07,
          7.95287878e-02,   1.25538743e-06,   9.21834882e-02,
         -9.77908062e-06,   6.06672265e-06,   4.74992196e-02,
          6.23011204e-07,  -5.72835047e-02,   1.15815898e-06,
          6.81089402e-02,  -8.13025747e-07,  -5.71773999e-04,
          1.29292049e-06,   1.80404640e-02,  -2.66561054e-07],
       [  2.79245151e-02,  -4.66264494e-01,  -2.76003881e-06,
         -1.78282306e-01,  -8.02399272e-07,  -1.57364102e-01,
          1.35962635e-05,  -3.34486458e-05,  -2.07061461e-01,
         -3.57480297e-05,   3.48571757e-01,  -1.24897169e-05,
         -1.61931814e+00,   2.05492488e-05,   1.27056841e-01,
         -7.98803943e-06,   5.54315647e-01,  -4.89896364e-05],
       [ -1.68392227e-08,  -2.88262349e-06,  -5.15548208e-01,
         -6.77141922e-07,   1.14910486e-07,  -3.97701075e-05,
         -4.08547057e-01,  -2.04428477e-01,  -1.84979519e-06,
         -2.17840914e-06,  -2.62824350e-05,  -9.90113099e-01,
          1.46937779e-05,  -1.05732121e-05,   1.84779379e-05,
          4.39099756e-06,   1.01227553e-05,  -3.71181941e-02],
       [ -1.64620661e-03,   1.22153835e-01,  -2.23241402e-06,
         -5.60013634e-01,  -9.55348349e-06,   2.66171614e-01,
         -1.87999444e-05,   7.51864717e-05,   5.25129623e-01,
         -6.01562857e-05,   8.10298150e-01,  -3.75404414e-05,
          1.94063327e-01,  -4.77354940e-06,  -6.32459877e-03,
          1.14904511e-05,  -4.04792653e-02,   3.48027368e-06],
       [  2.69212375e-08,   7.74649916e-07,  -4.01798077e-06,
         -1.37514955e-05,   6.43349353e-01,   3.31757598e-06,
          2.59845397e-06,   6.06568455e-06,  -6.43112871e-05,
         -9.60255577e-01,  -3.79231312e-05,   2.07769313e-06,
          2.00710963e-05,   6.08409369e-06,   2.64656156e-06,
          7.55943955e-03,  -5.86681107e-06,  -2.34699605e-06],
       [ -1.31929004e-02,  -4.67290191e-01,   7.14799797e-06,
         -3.35093433e-01,  -8.87458150e-06,  -1.12257251e+00,
          1.18348216e-04,   3.12411201e-05,   1.49843280e-01,
          4.52334964e-05,  -5.35970785e-03,   3.03014491e-06,
          2.53679260e+00,  -3.40597798e-05,  -2.53283751e-01,
          3.55350811e-06,  -1.32683498e+00,   1.03019788e-04],
       [  5.51033604e-08,   1.54544674e-06,  -2.63246869e-01,
         -4.42279964e-06,   5.50094699e-07,  -6.89375269e-05,
         -7.77183582e-01,  -4.62399451e-01,   9.33025539e-05,
         -2.28986100e-06,   6.31749975e-05,   1.67582364e+00,
         -2.27051777e-05,   8.83344358e-06,  -2.98862488e-05,
         -6.35865248e-06,   2.68109515e-05,   9.41973852e-01],
       [  2.42465963e-03,   6.02778891e-02,  -9.45253668e-07,
         -3.92809525e-01,  -9.00326871e-06,   4.38278374e-01,
         -5.03146899e-05,  -3.11010379e-05,  -1.24553342e-01,
          3.85425467e-05,  -1.16846976e+00,   4.30858050e-05,
         -6.38463748e-01,   2.03031683e-05,   1.41948847e-01,
         -3.86258304e-06,   6.73889644e-01,  -4.29282704e-05],
       [ -1.54502760e-08,   4.73377938e-08,  -7.80138135e-07,
         -8.87951281e-06,   5.07422362e-01,  -3.01379819e-06,
          1.48703409e-06,  -7.20693411e-06,   6.80348567e-05,
          1.03801079e+00,   3.86015006e-05,  -8.51695705e-07,
         -2.11782561e-05,  -1.68911503e-05,  -3.61578406e-06,
          3.32033750e-02,   4.58516371e-06,   1.42725993e-06],
       [  2.15691975e-04,   1.88275289e-02,   7.47453202e-07,
         -1.68323442e-02,   1.30729147e-08,   1.32403853e-02,
         -1.68890649e-06,  -1.49754450e-05,  -1.44449502e-01,
          3.75263807e-06,   6.18498090e-02,   6.82897122e-06,
          1.60749309e-01,  -1.07323615e-04,  -4.74643611e-01,
          1.86807845e-05,   9.80078470e-01,  -4.87449842e-05],
       [  1.07064751e-08,  -6.29526608e-07,   1.98841276e-06,
          5.97357809e-07,  -1.15290188e-06,  -1.33088194e-06,
          3.80974015e-06,   3.66738125e-06,  -4.60054451e-06,
         -8.80597564e-06,  -8.18950737e-07,   7.24200068e-06,
         -1.18344826e-06,  -9.99999957e-01,   1.98435094e-04,
         -2.15770728e-04,  -1.37923835e-05,   1.99847160e-06],
       [  8.88854749e-09,  -4.95196345e-07,   9.33325855e-07,
          7.90829637e-07,  -3.43726766e-02,  -2.85525197e-06,
          1.28918189e-06,   4.32965230e-07,  -9.21642673e-06,
         -1.51621477e-02,  -6.69302432e-06,   3.26389221e-06,
         -9.03994532e-06,  -2.15443523e-04,   7.24496700e-06,
          9.99294042e-01,  -1.48949623e-05,   2.46278358e-06],
       [ -1.24879353e-04,  -4.73849109e-03,  -1.18360441e-06,
         -3.02061107e-02,  -1.74076363e-06,   8.95644181e-03,
         -3.75153704e-06,   1.13548164e-05,   1.23125749e-01,
         -8.77043872e-06,  -8.12511699e-02,  -2.35094372e-05,
         -1.77955151e-01,  -1.66659584e-04,  -8.73794208e-01,
         -2.48149050e-06,  -5.10980247e-01,   2.02803430e-05],
       [ -3.62812548e-08,  -7.63004586e-07,   3.95128531e-02,
          1.22142101e-06,  -1.32388749e-06,  -6.36318459e-07,
          1.97778546e-02,  -2.06389767e-01,   2.71030588e-05,
          8.89626341e-07,   2.40345297e-06,   2.52153274e-02,
          1.83786681e-05,  -2.96627274e-06,   4.07667254e-06,
          3.05209698e-06,  -6.38598607e-05,  -1.27435733e+00],
       [  1.90437377e-04,  -1.39558423e-01,   2.41098414e-01,
          1.37606070e-01,   5.76576818e-06,   1.01222420e-01,
         -9.57146146e-02,  -8.47736228e-01,   8.13687947e-01,
         -5.56076860e-05,  -3.56373401e-01,   1.22589560e-03,
         -1.74427612e-01,   2.29699813e-06,   9.07805630e-02,
          1.26060760e-05,   8.28723502e-01,   9.55922031e-01],
       [  3.04380875e-03,  -1.01203508e-03,   1.34176528e-01,
          9.69307032e-02,   5.91806922e-06,   9.50397998e-01,
         -1.28965880e+00,   6.04496572e-01,  -5.61099128e-01,
          3.12605046e-05,  -1.49875195e-02,   9.35477743e-01,
         -6.68578508e-01,   1.41865740e-05,   4.64217192e-02,
         -8.44068071e-06,   1.10720006e-01,  -1.16701932e-02]])

    thresholds = {'ref_result_exp_alpha': 1e-08, 'ref_result_dm_alpha': 1e-08, 'ref_result_energy': 1e-08}

    test_path = context.get_fn("examples/hf_dft/rks_water_hybgga.py")

    l = {}
    with open(test_path) as fh:
        exec fh in l

    for k,v in thresholds.items():
        var_name = k.split("ref_")[1]
        assert allclose(l[var_name], l[k], v), l[k] - l[var_name]
