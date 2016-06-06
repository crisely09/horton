#!/usr/bin/env python

from __future__ import division
from math import exp, log
from scipy.optimize import minimize, optimize, minimize_scalar, leastsq, fmin_tnc




# Def constants
kb = 0.00000316689   # in Hartreee K


def occupation(x, y, z):
    """

    :param z:
    :param x:
    :param y:
    :return:
    """
    # x orbitals
    # y chemical potential
    # z theta

    if ((1 / (kb * z)) * (x - y)) < -700:
        f = 1
    elif ((1 / (kb * z)) * (x - y)) > 700:
        f = 0
    else:
        f = 1 / (1 + exp((1 / (kb * z)) * (x - y)))
    return f


def nE(x, y, z, ne):
    global dN, n
    """

    :param x:
    :param y:
    :param z:
    :return:
    """
    occlist = []
    for energy in x:
        occ = occupation(energy, y, z)
        occlist.append(occ)
    n = sum(occlist)
    dN = n - ne
    return {'n': n, 'dN': dN}


def entropy(x, y, z):
    global ent
    entropylist = []
    for energy in x:
        if occupation(energy, y, z) == 1:
            mi = 0
        else:
            mi = (1 - occupation(energy, y, z)) * log(1 - occupation(energy, y, z))

        if occupation(energy, y, z) == 0:
            pi = 0
        else:
            pi = (occupation(energy, y, z)) * log(occupation(energy, y, z))
        si = pi + mi
        entropylist.append(si)
    ent = sum(entropylist) * kb
    return ent


def energy_corrected(x, y, z, eDFA):
    # x are orbital energies
    # y is the chemical potential
    # l is the initial temperature
    # ehf is the energy computed at the low level of theory
    global  ecorr
    entropy(x, y, z)
    ecorr = eDFA + z * ent
    return ecorr


def temOptimizer(x, y, temper, eDFA, highE):
    # x are orbital energies
    # y is the chemical potential
    # temper is the initial temperature
    # ehf is the energy computed at the low level of theory
    # highE High Quality Energy

    def fitTemp(trialT):
        enerx = energy_corrected(x, y, trialT, eDFA)
        return enerx

    def errorT (trialT):
        errT = abs(fitTemp(trialT)-highE)
        return errT

    trial_t = temper
    rest = minimize(errorT, trial_t, method='Nelder-Mead', options={'xtol': 1e-7, 'disp': True})
    return rest.x


def muOptimized(x, mu, z, ne):
    # x are orbital energies
    # mu is the initial chemical potential
    # l is the  temperature
    # nE is the exact electrons numbers

    def fitMu(trialMu):
        occlistFit = []
        for energy in x:
            occf = occupation(energy, trialMu, z)
            occlistFit.append(occf)
        return sum(occlistFit)

    def errorN(trialMu):
        err = abs(ne - fitMu(trialMu))
        return err

    trial_0 = mu
    res = minimize(errorN, trial_0, method='Nelder-Mead', options={'xtol': 1e-6, 'disp': True})
    return res.x


def mu_thetaOpt(x, y, guess_theta, ne, eDFA, highE):
    # x are orbital energies
    # y is the chemical potential
    # guess_theta is the initial temperature
    # nE is the exact electrons numbers

    """

    :rtype: float
    """
    global muOpt, thetOpt
    tolerance = 0.00000001
    resultsTemp = [10]
    guess = [guess_theta]
    thets = [guess[len(guess) - 1]]
    mus = [y]

    for i in resultsTemp:
        if i > tolerance:
            opt1 = temOptimizer(x, mus[len(mus) - 1], thets[len(thets) - 1], eDFA, highE)
            thets.append(opt1)

            opt2 = muOptimized(x, mus[len(mus) - 1], thets[len(thets) - 1], ne)
            mus.append(opt2)

            nE(x, opt2, opt1, ne)
            print dN, ne


            resultsTemp_i = abs(energy_corrected(x, opt2, opt1, eDFA) - highE)  # nElectrons
            print resultsTemp_i

            resultsTemp.append(resultsTemp_i)

        else:
           continue

    muOpt = mus[len(mus) - 1]
    thetOpt = thets[len(thets) - 1]
    return {'muOpt': muOpt, 'thetOpt': thetOpt}



def occupationsOptimizer(orbitals, muTrial, tempTrial, ne, eDFA, highE):
    # x are orbital energies
    # cp is the chemical potential
    # ne is the electrons number
    # th is the temperature

    global  newOccupations
    results = []
    newOccupations = []

    mu_thetaOpt(orbitals, muTrial, tempTrial, ne, eDFA, highE)
    nE(orbitals, muOpt, thetOpt, ne)
    e = entropy(orbitals, muOpt, thetOpt)
    entropy_per_electron = e / ne


    results.append(float(ne))
    results.append(float(muOpt))
    results.append(float(thetOpt))
    results.append(float(dN))
    results.append(float(abs(e)))
    results.append(float(e*thetOpt))
    results.append(float(abs(entropy_per_electron)))

    energy_corrected(orbitals, muOpt, thetOpt, eDFA)
    print ecorr

    for energy in orbitals:
        ocn = occupation(energy, muOpt, thetOpt)
        newOccupations.append(ocn)

    def saveResults():
        file = open('data2.term', 'a')
        file.write("%s \n\n" % results)
        file.close()
    saveResults()

    def saveEntropPerElec():
        file = open('entpe.term', 'a')
        file.write("%s \n" % entropy_per_electron)
        file.close()
    saveEntropPerElec()

    return {'mu': float(muOpt), 'temp': float(thetOpt), 'newOccupations': newOccupations}
