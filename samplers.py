r""" MCMC SAMPLERS

PURPOSE: 
# ------------------------------------- . ------------------------------------ #
FUNCTIONS:
ma    : performs the Metropolis algorithm (symmetric proposal dist.)
mha   : performs the Metropolis-Hastings algorithm (asymmetric proposal dist.)
gibbs : performs the Gibbs sampling
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


def ma(x0, target, proposal, spread, N, burnIn, lag):

    if proposal == 'normal':
        q = st.norm(loc=0, scale=spread)
    elif proposal == 'uniform':
        q = st.uniform(loc=0-spread, scale=2*spread)
    else:



    samples = np.zeros(N)
    samples[0] = x0  # Initial seed for the chain
    acc = 0  # Count the number of accepted samples

    for i in range(N - 1):
        xstar = proposal(samples[i])  # Candidate sample from a proposal PDF
        alpha = min(target(xstar)/target(samples[i]), 1)
        u = st.uniform.rvs(loc=0, scale=1)

        if u <= alpha:
            samples[i + 1] = xstar  # Take the candidate
            acc += 1
        else:
            samples[i + 1] = samples[i]  # Do not take the candidate

    return samples, acc/N


def mha(target, proposalrvs, proposalpdf, x1, N):

    samples = np.zeros(N)
    samples[0] = x1  # Initial seed for the chain
    acc = 0  # Count the number of accepted samples

    for i in range(N - 1):
        # xstar = Candidate sample from a proposal PDF
        xstar = proposalrvs(samples[i])
        # c = correction factor of the asymmetric proposal distribution MHA
        c = proposalpdf(samples[i], xstar)/proposalpdf(xstar, samples[i])
        alpha = min(c*target(xstar)/target(samples[i]), 1)
        u = st.uniform.rvs(loc=0, scale=1)

        if u <= alpha:
            samples[i + 1] = xstar  # Take the candidate
            acc += 1
        else:
            samples[i + 1] = samples[i]  # Do not take the candidate
    return samples, acc/N