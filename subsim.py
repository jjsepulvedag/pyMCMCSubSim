r"""SUBSET SIMULATION ALGORITHM

PURPOSE: Implementation of the Subset Simulation Algorithm
# ------------------------------------- . ------------------------------------ #
FUNCTIONS:
ssa    : SubSet Simulation Algorithm
# ------------------------------------- . ------------------------------------ #
AUTHORSHIP
Written by: Juan Jose Sepulveda Garcia
Mail      : jjs134@uclive.ac.nz / jjsepulvedag@unal.edu.co
Date      : March 2025
# ------------------------------------- . ------------------------------------ #
REFERENCES:
NA
"""

import numpy as np
import scipy.stats as st
import mcmc as mcmc

def ssa(pi, pi_marginal, g_lim, p0=0.1, N=1000, spread=1, g_failure='>=0'):
    ''' Subset simulation
    Usage:
        pf = subsim(pi, pi_marginal, g_lim, N=1000, p0=0.1)

    Input parameters:
        pi:            JPDF of theta
        pi_marginal:   list with the marginal PDFs associated to the PDF pi
        g_lim:         limit state function g(x)
        N:   number of samples per conditional level, default N=1000
        p0:  conditional failure probability p0 in [0.1, 0.3], default p0=0.1
        g_failure:     definition of the failure region; the possible options
                       are: >=0' (default) and '<=0'

    Output parameters:
        theta:  list of samples for each intermediate failure level
        g:      list of the evaluations in g_lim of the samples of theta
        b:      intermediate failure thresholds
        pf:     probability of failure
    '''
    if g_failure == '>=0':
        sign_g = 1
    elif g_failure == '<=0':
        sign_g = -1
    else:
        raise Exception("g_failure must be either '>=0' or '<=0'.")

    d = len(pi_marginal)  # number of dimensions of the random variable theta

    Nc = int(N*p0)  # number of Markov chains (number of seeds per level)
    Ns = int(1/p0)  # number of samples per Markov chain, including the seed

    if not (np.isclose(Nc, N*p0) and np.isclose(Ns, 1/p0)):
        raise Exception("Please choose p0 so that N*p0 and 1/p0 are natural "
                        "numbers")

    # INITIALIZATION OF SOME LISTS
    N_F   = []  # N_F[j] contains the number of failure samples at level j
    theta = []  # list of numpy arrays which contains the samples at each level j
    g     = []  # list of numpy arrays which contains the evaluation of each set
                # of samples at each level j
    b     = []  # intermediate threshold values

    # CRUDE MONTE CARLO IN REGION F[0]
    j = 0       # number of conditional level

    # draw N i.i.d. samples from pi = pi(.|F0) using MCS
    theta.append(None)
    theta[0] = pi.rvs(N)
    if d == 1:
        theta[0] = theta[0][:, np.newaxis]

    # evaluate the limit state function of those N samples
    g.append(None)
    g[0] = np.empty(N)
    for i in range(N):
        g[0][i] = sign_g*g_lim(theta[0][i, :])

    # count the number of samples in level F[0]
    N_F.append(None)
    N_F[0] = np.sum(g[0] > 0)  # b = 0

    # MAIN LOOP
    while N_F[j]/N < p0:  # if N_F[j] < Nc
        # sort the limit state values in ascending order
        idx = np.argsort(g[j]) # index of the sorting
        g_sorted = g[j][idx]   # np.sort(g) -> sort the points using the idx key

        # estimate the p0-percentile of ghttps://www.anaconda.com/distribution/
        b.append(None)
        b[j] = (g_sorted[N-Nc-1] + g_sorted[N-Nc])/2
        #b[j] = (g_sorted[-Nc-1] + g_sorted[-Nc])/2
        #b[j] = np.percentile(g_sorted, 100*(1-p0))

        # select the seeds: they are the last Nc samples associated to idx
        seed  = theta[j][idx[-Nc:], :]
        gseed = g[j][idx[-Nc:]]

        # starting from seed[k,:] draw Ns-1 additional samples from pi(.|Fj)
        # using a MCMC algorithm called MMA
        theta_from_seed = Nc*[None]
        g_from_seed     = Nc*[None]
        num_eval_g      = Nc*[None]
        ar_rate         = Nc*[None]
        for k in range(Nc): # Nc = N*p0
            theta_from_seed[k], g_from_seed[k], num_eval_g[k], ar_rate[k] = \
                mcmc.MMA(seed[k,:], gseed[k], Ns, g_lim, b[j], pi_marginal, 
                         spread, g_failure)

        # concatenate all samples theta_from_seed[k] in a single array theta
        theta.append(None);  theta[j+1] = np.vstack(theta_from_seed)
        g.append(None);      g[j+1]     = sign_g*np.concatenate(g_from_seed)

        # count the number of samples in level F[j+1]
        N_F.append(None)
        N_F[j+1] = np.sum(g[j+1] > 0) # b = 0

        # continue with the next intermediate failure level
        j += 1

    # %% estimate the probability of failure and report it
    pf = p0**j * N_F[j]/N

    # change of sign for g
    if g_failure == '<=0':  # sign_g == -1
        for i in range(len(g)):   g[i] = -g[i]
        for i in range(len(g)-1): b[i] = -b[i]

    return theta, g, b, pf


if __name__ == '__main__':
    print('JJSG: Hello world!')