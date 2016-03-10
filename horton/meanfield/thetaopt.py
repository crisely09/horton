#!/usr/bin/env python
from __future__ import division
import sympy
from numpy import sign
from math import exp, tanh, log
from scipy.optimize import minimize


# Data required
# Compounds to be analyzed. Must match with the number of inputs. .36

# functions for this code
# noinspection PyArgumentEqualDefault

__all__ = ['optimizedOccupations',
]

def sech(argument):
    """

    :param argument:
    :return:
    """
    return sympy.cosh(argument) ** (-1)


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

    if ((1 / z) * (x - y)) < -700:
        f = 1
    elif ((1 / z) * (x - y)) > 700:
        f = 0.00000000000000001
    else:
        f = 1 / (1 + exp((1 / z) * (x - y)))
    return f


def shanonocc(x, y, z):
    if ((1 / z) * (x - y)) < -35 or ((1 / z) * (x - y)) > 35:
        socc = 0
    else:
        socc = (1 / (1 + exp((1 / z) * (x - y)))) * log((1 / (1 + exp((1 / z) * (x - y))))) + (1 - (
            1 / (1 + exp((1 / z) * (x - y))))) * log(1 - (1 / (1 + exp((1 / z) * (x - y)))))
    return socc


def orbSoftness(x, y, z):
    """

    :param x:
    :param y:
    :param z:
    :return:
    """
    return (1 / 4) * (sech((1 / (2 * z)) * (y - x))) ** 2


def orbhyperSoftness(x, y, z):
    """

    :param x:
    :param y:
    :param z:
    :return:
    """
    return (1 / 4) * ((sech((1 / (2 * z)) * (y - x))) ** 2) * tanh((1 / (2 * z)) * (y - x))


def orbsuperSoftness(x, y, z):
    """

    :rtype: float
    :param x:
    :param y:
    :param z:
    :return:
    """
    return 2 * ((orbSoftness(x, y, z)) ** 2) - orbhyperSoftness(x, y, z) * tanh((1 / (2 * z)) * (y - x))


def nE(x, y, z, ne):############################################################
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


def occderivatives(x, y, z):
    global s_fi, rPrime, sPrime, sBiprime, sBiprime, tot_s, sTriprime

    s = []
    se = []
    see = []
    h = []
    he = []
    hee = []
    heee = []
    s_i = []
    u = []
    ue = []
    uee = []
    ueee = []
    ueeee = []

    for energy in x:
        s_ = orbSoftness(energy, y, z)
        se_ = orbSoftness(energy, y, z) * energy
        see_ = orbSoftness(energy, y, z) * (energy ** 2)
        h_ = orbhyperSoftness(energy, y, z)
        he_ = orbhyperSoftness(energy, y, z) * energy
        hee_ = orbhyperSoftness(energy, y, z) * (energy ** 2)
        heee_ = orbhyperSoftness(energy, y, z) * (energy ** 3)
        u_ = orbsuperSoftness(energy, y, z)
        ue_ = orbsuperSoftness(energy, y, z) * energy
        uee_ = orbsuperSoftness(energy, y, z) * (energy ** 2)
        ueee_ = orbsuperSoftness(energy, y, z) * (energy ** 3)
        ueeee_ = orbsuperSoftness(energy, y, z) * (energy ** 4)
        si_ = log(occupation(energy,y,z))+(1-occupation(energy,y,z))*(energy-y)/z
        # si_ = (1-occupation(energy,y,z))*(energy-y)+ z * log(occupation(energy,y,z))
        #si_ = z * shanonocc(energy, y, z)
        s.append(s_)
        se.append(se_)
        see.append(see_)
        h.append(h_)
        he.append(he_)
        hee.append(hee_)
        heee.append(heee_)
        # "Third derivative parameters"
        u.append(u_)
        ue.append(ue_)
        uee.append(uee_)
        ueee.append(ueee_)
        ueeee.append(ueeee_)
        s_i.append(si_)
    tot_s = float(sum(s))
    tot_se = float(sum(se))
    tot_see = float(sum(see))
    tot_h = float(sum(h))
    tot_he = float(sum(he))
    tot_hee = float(sum(hee))
    tot_heee = float(sum(heee))
    # "Third derivative parameters"
    tot_u = float(sum(u))
    tot_ue = float(sum(ue))
    tot_uee = float(sum(uee))
    tot_ueee = float(sum(ueee))
    tot_ueeee = float(sum(ueeee))
    s_fi = float(sum(s_i))

    #try:
     #   a = tot_se / tot_s
    #except:
       # a = 1000000

    R  = tot_se / tot_s
    rPrime = 2 * R * tot_he - tot_hee - (R ** 2) * tot_h
    sPrime = (R * tot_se - tot_see)
    sBiprime = (-(R ** 3) * tot_h + 3 * (R ** 2) * tot_he - 3 * R * tot_hee + tot_heee)
    sTriprime = -tot_ueeee + 4 * R * tot_ueee - 6 * (R ** 2) * tot_uee + 4 * (R ** 3) * tot_ue - (R ** 4) * tot_u
    return {'s_fi': s_fi, 'rPrime': rPrime, 'sPrime': sPrime, 'sBiprime': sBiprime,'tot_s': tot_s, 'sTriprime': sTriprime}


def energy_correction(z):
    global de
    de = s_fi * z
    return de


def first_derivative(z):
    s1d = (1 / (z ** 2)) * sPrime + s_fi
    return s1d


def second_derivative(z):
    global s2d
    s2d = -(1 / (z ** 3)) * sPrime + (1 / ((z ** 4))) * sBiprime
    return s2d


def third_derivative(z):
    s3d = (3 / (z ** 4)) * sPrime - (5 / ((z ** 5))) * sBiprime + (3 / (tot_s * (z ** 6))) * (rPrime) ** 2 + (1 / (
        z ** 6)) * sTriprime
    return s3d


def muOptimized(x, guess_mu, z, ne):  ################################
    def fitMu(trial):
        occlistFit = []
        for energy in x:
            occf = occupation(energy, trial, z)
            occlistFit.append(occf)
        return sum(occlistFit)

    def error(trial):
        err = abs(ne - fitMu(trial))
        return err

    trial_0 = guess_mu
    res = minimize(error, trial_0, method='Nelder-Mead', options={'xtol': 1e-6, 'disp': True})
    return res.x


def thetaOptimized(x, y, l):
    # x are orbital energies
    # y is the chemical potential
    # l is the trial temperature

    # def occ(x,y,w):
    #    prop=occderivatives(x, y, w)
    #    return prop


    def fitTheta(w):
        occderivatives(x, y, w)
        a = - first_derivative(w)
        return a

    def derivf(w):
        occderivatives(x, y, w)
        b = - second_derivative(w)
        return b

    def second__derivf(w):
        occderivatives(x, y, w)
        c = - third_derivative(w)
        return c

    wo = l


    #res = minimize(derivf, wo, method='Nelder-Mead', options={'disp': True, 'xtol': 5e-5, 'maxiter': 60})
    res = minimize(fitTheta, wo, method='Newton-CG', jac=derivf, hess=second__derivf, options={'disp': True,'xtol': 1e-5, 'maxiter':100})

    print res.x
    newTheta = res.x
    return newTheta


def mu_thetaOpt(x, y, guess_theta, ne): #############################
    """

    :rtype: float
    """
    global muOpt, thetOpt
    tolerance = 0.0005
    resultsmu = [10]
    errors = [10, 20, 30]
    # iters = [0]
    guess = [guess_theta]
    thets = [guess[len(guess) - 1]]
    mus = [y]

    for i in resultsmu:
        if abs(i) > tolerance:
            opt1 = muOptimized(x, mus[len(mus) - 1], thets[len(thets) - 1], ne)
            mus.append(opt1)

            opt2 = thetaOptimized(x, mus[len(mus) - 1], thets[len(thets) - 1])
            thets.append(opt2)

            occlist = []

            for energy in x:
                occ = occupation(energy, opt1, opt2)
                occlist.append(occ)
            print 'occs', sum(occlist)
            resultmu_i = ne - sum(occlist)  # nElectrons

            p1 = errors[len(errors) - int(resultsmu[len(resultsmu) - 1] / i)]
            p2 = errors[len(errors) - int(resultsmu[len(resultsmu) - 1] / i) - 1]
            p3 = errors[len(errors) - int(resultsmu[len(resultsmu) - 1] / i) - 2]
            p4 = errors[len(errors) - int(resultsmu[len(resultsmu) - 1] / i) - 4]

            resultsmu.append(resultmu_i)
            errors.append(resultmu_i)
            print 'errors', errors

            if abs(p1) > 0.99999 and abs(p1 - p2) < 0.00001 and abs(p1 - p3) < 0.00001 and abs(p1-p4) < 0.00001:
                a = guess[len(guess) - 1] + 0.2
                thets.append(a)
                guess.append(a)
                mus.append(y)
                errors.append(errors[len(errors) - 1] + 0.5)
                print guess
            else:
                continue

    muOpt = mus[len(mus) - 1]
    thetOpt = thets[len(thets) - 1]
    return {'muOpt': muOpt, 'thetOpt': thetOpt}


def optimizedOccupations(x, cp, ne, th):
    global  newOccupations
    results=[]
    newOccupations = []
    theta = th
    fermisE = cp
    mu_thetaOpt(x, fermisE, theta, ne)
    nE(x, muOpt, thetOpt, ne)
    occderivatives(x, muOpt, thetOpt)
    energy_correction(thetOpt)
    second_derivative(thetOpt)
    results.append(float(muOpt))
    results.append(float(thetOpt))
    results.append(float(de))
    results.append(float(s2d))
    results.append(float(dN))
    for energy in x:
        ocn = occupation(energy, muOpt, thetOpt)
        newOccupations.append(ocn)
    def saveResults():
        file = open('data1.term', 'a')
        file.write("%s \n\n" % results)
        file.close()
    saveResults()


    return float(muOpt), float(thetOpt), newOccupations
