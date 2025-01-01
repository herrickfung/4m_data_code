'''
for all function,
stim, resp, conf receives array
'''

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import Bounds, LinearConstraint, minimize, SR1


def compute_sdt(stim, resp):
    s1 = (stim == 1).sum()
    s2 = (stim == 0).sum()

    hit_count = ((stim == 1) & (resp == 1)).sum()
    fa_count = ((stim == 0) & (resp == 1)).sum()

    hit_rate = hit_count / s1
    fa_rate = fa_count / s2

    if hit_rate == 0:
        hit_rate = 0.5 / s1
    elif hit_rate == 1:
        hit_rate = 1 - 0.5 / s1

    if fa_rate == 0:
        fa_rate = 0.5 / s2
    elif fa_rate == 1:
        fa_rate = 1 - 0.5 / s2

    Z_hit = norm.ppf(hit_rate)
    Z_fa = norm.ppf(fa_rate)

    d_prime = Z_hit - Z_fa
    c = -0.5 * (Z_hit + Z_fa)
    beta = np.exp(d_prime * c)

    return d_prime, c, beta


def compute_sdt_criteria(stim, resp, conf, n_ratings):
    HR = np.zeros(shape=(2*n_ratings-1))
    FAR = np.zeros(shape=(2*n_ratings-1))
    for roc in range(1, n_ratings):
        HR[roc-1] = np.sum((stim == np.max(stim)) & ((resp == np.max(stim)) | (conf <= (n_ratings - roc)))) / np.sum(stim == np.max(stim))
        FAR[roc-1] = np.sum((stim == np.min(stim)) & ((resp == np.max(stim)) | (conf <= (n_ratings - roc)))) / np.sum(stim == np.min(stim))

    for roc in range(n_ratings, 2*n_ratings):
        HR[roc-1] = np.sum((stim == np.max(stim)) & ((resp == np.max(stim)) & (conf > (roc - n_ratings)))) / np.sum(stim == np.max(stim))
        FAR[roc-1] = np.sum((stim == np.min(stim)) & ((resp == np.max(stim)) & (conf > (roc - n_ratings)))) / np.sum(stim == np.min(stim))

    dprime = norm.ppf(HR[n_ratings-1]) - norm.ppf(FAR[n_ratings-1])
    criterion = -0.5 * (norm.ppf(HR) + norm.ppf(FAR))
    return dprime, criterion


def compute_meta_count(stim, resp, conf, n_ratings, pad_cells, pad_amount):
    # check for valid inputs
    if not (len(stim) == len(resp) == len(conf)):
        raise('stimID, response, and rating input vectors must have the same lengths')

    ''' filter bad trials '''
    valid_trials = ((stim == 0) | (stim == 1)) & \
                   ((resp == 0) | (resp == 1)) & \
                   ((conf >= 1) | (conf <= n_ratings))
    stim, resp, conf = stim[valid_trials], resp[valid_trials], conf[valid_trials]

    if pad_amount == None:
        pad_amount = 1/(2*n_ratings)

    nR_S1 = np.zeros(n_ratings*2)
    nR_S2 = np.zeros(n_ratings*2)

    for r in range(n_ratings):
        nR_S1[r] = np.sum((stim == 0) & (resp == 0) & (conf == n_ratings-r))
        nR_S2[r] = np.sum((stim == 1) & (resp == 0) & (conf == n_ratings-r))

    for r in range(n_ratings):
        nR_S1[r+n_ratings] = np.sum((stim == 0) & (resp == 1) & (conf == r+1))
        nR_S2[r+n_ratings] = np.sum((stim == 1) & (resp == 1) & (conf == r+1))

    if pad_cells:
        nR_S1 += pad_amount
        nR_S2 += pad_amount

    return nR_S1, nR_S2


# returns negative log-likelihood of parameters given experimental data
# parameters[0] = meta d'
# parameters[1:end] = type-2 criteria locations
def fit_meta_d_logL(parameters,inputObj):
    meta_d1 = parameters[0]
    t2c1    = parameters[1:]
    nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv = inputObj

    # define mean and SD of S1 and S2 distributions
    S1mu = -meta_d1/2
    S1sd = 1
    S2mu = meta_d1/2
    S2sd = S1sd/s

    # adjust so that the type 1 criterion is set at 0
    # (this is just to work with optimization toolbox constraints...
    #  to simplify defining the upper and lower bounds of type 2 criteria)
    S1mu = S1mu - eval(constant_criterion)
    S2mu = S2mu - eval(constant_criterion)

    t1c1 = 0

    # set up MLE analysis
    # get type 2 response counts
    # S1 responses
    nC_rS1 = [nR_S1[i] for i in range(nRatings)]
    nI_rS1 = [nR_S2[i] for i in range(nRatings)]
    # S2 responses
    nC_rS2 = [nR_S2[i+nRatings] for i in range(nRatings)]
    nI_rS2 = [nR_S1[i+nRatings] for i in range(nRatings)]

    # get type 2 probabilities
    C_area_rS1 = fncdf(t1c1,S1mu,S1sd)
    I_area_rS1 = fncdf(t1c1,S2mu,S2sd)

    C_area_rS2 = 1-fncdf(t1c1,S2mu,S2sd)
    I_area_rS2 = 1-fncdf(t1c1,S1mu,S1sd)

    t2c1x = [-np.inf]
    t2c1x.extend(t2c1[0:(nRatings-1)])
    t2c1x.append(t1c1)
    t2c1x.extend(t2c1[(nRatings-1):])
    t2c1x.append(np.inf)

    prC_rS1 = [( fncdf(t2c1x[i+1],S1mu,S1sd) - fncdf(t2c1x[i],S1mu,S1sd) ) / C_area_rS1 for i in range(nRatings)]
    prI_rS1 = [( fncdf(t2c1x[i+1],S2mu,S2sd) - fncdf(t2c1x[i],S2mu,S2sd) ) / I_area_rS1 for i in range(nRatings)]

    prC_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S2mu,S2sd)) - (1-fncdf(t2c1x[nRatings+i+1],S2mu,S2sd)) ) / C_area_rS2 for i in range(nRatings)]
    prI_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S1mu,S1sd)) - (1-fncdf(t2c1x[nRatings+i+1],S1mu,S1sd)) ) / I_area_rS2 for i in range(nRatings)]

    # calculate logL
    logL = np.sum([
            nC_rS1[i]*np.log(prC_rS1[i]) \
            + nI_rS1[i]*np.log(prI_rS1[i]) \
            + nC_rS2[i]*np.log(prC_rS2[i]) \
            + nI_rS2[i]*np.log(prI_rS2[i]) for i in range(nRatings)])

    if np.isinf(logL) or np.isnan(logL):
#        logL=-np.inf
        logL=-1e+300 # returning "-inf" may cause optimize.minimize() to fail
    return -logL


def fit_meta_d_MLE(nR_S1, nR_S2, s = 1+1e-7, fncdf = norm.cdf, fninv = norm.ppf):

    # check inputs
    if (len(nR_S1) % 2)!=0:
        raise('input arrays must have an even number of elements')
    if len(nR_S1)!=len(nR_S2):
        raise('input arrays must have the same number of elements')
    if any(np.array(nR_S1) == 0) or any(np.array(nR_S2) == 0):
        print(' ')
        print('WARNING!!')
        print('---------')
        print('Your inputs')
        print(' ')
        print('nR_S1:')
        print(nR_S1)
        print('nR_S2:')
        print(nR_S2)
        print(' ')
        print('contain zeros! This may interfere with proper estimation of meta-d''.')
        print('See ''help fit_meta_d_MLE'' for more information.')
        print(' ')
        print(' ')

    nRatings = int(len(nR_S1) / 2)  # number of ratings in the experiment
    nCriteria = int(2*nRatings - 1) # number criteria to be fitted

    """
    set up constraints for scipy.optimize.minimum()
    """
    # parameters
    # meta-d' - 1
    # t2c     - nCriteria-1
    # constrain type 2 criteria values,
    # such that t2c(i) is always <= t2c(i+1)
    # want t2c(i)   <= t2c(i+1)
    # -->  t2c(i+1) >= t2c(i) + 1e-5 (i.e. very small deviation from equality)
    # -->  t2c(i) - t2c(i+1) <= -1e-5
    A = []
    ub = []
    lb = []
    for ii in range(nCriteria-2):
        tempArow = []
        tempArow.extend(np.zeros(ii+1))
        tempArow.extend([1, -1])
        tempArow.extend(np.zeros((nCriteria-2)-ii-1))
        A.append(tempArow)
        ub.append(-1e-5)
        lb.append(-np.inf)

    # lower bounds on parameters
    LB = []
    LB.append(-10.)                              # meta-d'
    LB.extend(-20*np.ones((nCriteria-1)//2))    # criteria lower than t1c
    LB.extend(np.zeros((nCriteria-1)//2))       # criteria higher than t1c

    # upper bounds on parameters
    UB = []
    UB.append(10.)                           # meta-d'
    UB.extend(np.zeros((nCriteria-1)//2))      # criteria lower than t1c
    UB.extend(20*np.ones((nCriteria-1)//2))    # criteria higher than t1c

    """
    prepare other inputs for scipy.optimize.minimum()
    """
    # select constant criterion type
    constant_criterion = 'meta_d1 * (t1c1 / d1)' # relative criterion

    # set up initial guess at parameter values
    ratingHR  = []
    ratingFAR = []
    for c in range(1,int(nRatings*2)):
        ratingHR.append(sum(nR_S2[c:]) / sum(nR_S2))
        ratingFAR.append(sum(nR_S1[c:]) / sum(nR_S1))

    # obtain index in the criteria array to mark Type I and Type II criteria
    t1_index = nRatings-1
    t2_index = list(set(list(range(0,2*nRatings-1))) - set([t1_index]))

    d1 = (1/s) * fninv( ratingHR[t1_index] ) - fninv( ratingFAR[t1_index] )
    meta_d1 = d1

    c1 = (-1/(1+s)) * ( fninv( ratingHR ) + fninv( ratingFAR ) )
    t1c1 = c1[t1_index]
    t2c1 = c1[t2_index]

    # initial values for the minimization function
    guess = [meta_d1]
    guess.extend(list(t2c1 - eval(constant_criterion)))

    # other inputs for the minimization function
    inputObj = [nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv]
    bounds = Bounds(LB,UB)
    linear_constraint = LinearConstraint(A,lb,ub)

    # minimization of negative log-likelihood
    results = minimize(fit_meta_d_logL, guess, args = (inputObj), method='trust-constr',
                       jac='2-point', hess=SR1(),
                       constraints = [linear_constraint],
                       options = {'verbose': 1}, bounds = bounds)

    # quickly process some of the output
    meta_d1 = results.x[0]
    t2c1    = results.x[1:] + eval(constant_criterion)
    logL    = -results.fun

    # data is fit, now to package it...
    # find observed t2FAR and t2HR

    # I_nR and C_nR are rating trial counts for incorrect and correct trials
    # element i corresponds to # (in)correct w/ rating i
    I_nR_rS2 = nR_S1[nRatings:]
    I_nR_rS1 = list(np.flip(nR_S2[0:nRatings],axis=0))

    C_nR_rS2 = nR_S2[nRatings:];
    C_nR_rS1 = list(np.flip(nR_S1[0:nRatings],axis=0))

    obs_FAR2_rS2 = [sum( I_nR_rS2[(i+1):] ) / sum(I_nR_rS2) for i in range(nRatings-1)]
    obs_HR2_rS2 = [sum( C_nR_rS2[(i+1):] ) / sum(C_nR_rS2) for i in range(nRatings-1)]
    obs_FAR2_rS1 = [sum( I_nR_rS1[(i+1):] ) / sum(I_nR_rS1) for i in range(nRatings-1)]
    obs_HR2_rS1 = [sum( C_nR_rS1[(i+1):] ) / sum(C_nR_rS1) for i in range(nRatings-1)]

    # find estimated t2FAR and t2HR
    S1mu = -meta_d1/2
    S1sd = 1
    S2mu =  meta_d1/2
    S2sd = S1sd/s;

    mt1c1 = eval(constant_criterion)

    C_area_rS2 = 1-fncdf(mt1c1,S2mu,S2sd)
    I_area_rS2 = 1-fncdf(mt1c1,S1mu,S1sd)

    C_area_rS1 = fncdf(mt1c1,S1mu,S1sd)
    I_area_rS1 = fncdf(mt1c1,S2mu,S2sd)

    est_FAR2_rS2 = []
    est_HR2_rS2 = []

    est_FAR2_rS1 = []
    est_HR2_rS1 = []


    for i in range(nRatings-1):

        t2c1_lower = t2c1[(nRatings-1)-(i+1)]
        t2c1_upper = t2c1[(nRatings-1)+i]

        I_FAR_area_rS2 = 1-fncdf(t2c1_upper,S1mu,S1sd)
        C_HR_area_rS2  = 1-fncdf(t2c1_upper,S2mu,S2sd)

        I_FAR_area_rS1 = fncdf(t2c1_lower,S2mu,S2sd)
        C_HR_area_rS1  = fncdf(t2c1_lower,S1mu,S1sd)

        est_FAR2_rS2.append(I_FAR_area_rS2 / I_area_rS2)
        est_HR2_rS2.append(C_HR_area_rS2 / C_area_rS2)

        est_FAR2_rS1.append(I_FAR_area_rS1 / I_area_rS1)
        est_HR2_rS1.append(C_HR_area_rS1 / C_area_rS1)


    # package output
    fit = {}
    fit['da']       = np.sqrt(2/(1+s**2)) * s * d1

    fit['s']        = s

    fit['meta_da']  = np.sqrt(2/(1+s**2)) * s * meta_d1

    fit['M_diff']   = fit['meta_da'] - fit['da']

    fit['M_ratio']  = fit['meta_da'] / fit['da']

    mt1c1         = eval(constant_criterion)
    fit['meta_ca']  = ( np.sqrt(2)*s / np.sqrt(1+s**2) ) * mt1c1

    t2ca          = ( np.sqrt(2)*s / np.sqrt(1+s**2) ) * np.array(t2c1)
    fit['t2ca_rS1']     = t2ca[0:nRatings-1]
    fit['t2ca_rS2']     = t2ca[(nRatings-1):]

    fit['S1units'] = {}
    fit['S1units']['d1']        = d1
    fit['S1units']['meta_d1']   = meta_d1
    fit['S1units']['s']         = s
    fit['S1units']['meta_c1']   = mt1c1
    fit['S1units']['t2c1_rS1']  = t2c1[0:nRatings-1]
    fit['S1units']['t2c1_rS2']  = t2c1[(nRatings-1):]

    fit['logL']    = logL

    fit['est_HR2_rS1']  = est_HR2_rS1
    fit['obs_HR2_rS1']  = obs_HR2_rS1

    fit['est_FAR2_rS1'] = est_FAR2_rS1
    fit['obs_FAR2_rS1'] = obs_FAR2_rS1

    fit['est_HR2_rS2']  = est_HR2_rS2
    fit['obs_HR2_rS2']  = obs_HR2_rS2

    fit['est_FAR2_rS2'] = est_FAR2_rS2
    fit['obs_FAR2_rS2'] = obs_FAR2_rS2

    return fit


def compute_meta_sdt(stim, resp, conf, n_ratings, pad_cells=True, pad_amount=None):
    nr_s1, nr_s2 = compute_meta_count(stim, resp, conf, n_ratings, pad_cells, pad_amount)
    result = fit_meta_d_MLE(nr_s1, nr_s2)
    return result


def compute_folded_X(stim, resp, conf):
    c_conf = np.mean(conf[stim == resp])
    ic_conf = np.mean(conf[stim != resp])
    return c_conf, ic_conf


