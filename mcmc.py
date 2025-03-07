r"""MARKOV CHAIN MONTE CARLO ALGORITHMS

PURPOSE : Implementation of common Markov Chain Monte Carlo Algorithms
# ------------------------------------- . ------------------------------------ #
FUNCTIONS:
ma       : Metropolis algorithm
mha      : Metropolis-Hastings algorithm 
gibbs    : Gibbs sampler algorithm 
mma      : Modified Metropolis Algorithm (to be used with SubSim)
# ------------------------------------- . ------------------------------------ #
AUTHORSHIP
Written by: Juan Jose Sepulveda Garcia
Mail      : jjs134@uclive.ac.nz / jjsepulvedag@unal.edu.co 
Date      : March 2025
# ------------------------------------- . ------------------------------------ #
REFERENCES:
NA

NOTES:

"""
import numpy as np
import scipy.stats as st


def ma(x0, d, targetDist, proposalDist, N, burnIn, lag):
    '''METROPOLIS ALGORITHM
    Input 
        d        = dimension 
        x0       = initial sample
        target   = target function
        proposal = proposal function. IT MUST BE A SYMMETRICAL DISTRIBUTION!!!
        N        = desired number of samples
        burnIn   = number of samples for the burnIn period
        lag      = number of samples for the lag period
    Output:
        fsamples = samples obtained from the Markov Chain
        acc/NT   = acceptance rate
    '''

    Nt = burnIn + N*lag # Total number of samples I need to generate in order 
                        # to get N samples considering burnIn and Lag periods
    samples = np.zeros((Nt, d))
    samples[0, :] = x0  # Initial seed for the chain
    acc = 0  # Count the number of accepted samples

    for i in range(Nt - 1):
        xstar = proposalDist(samples[i])  # Drawing samples from a proposal PDF
        alpha = min(targetDist(xstar)/targetDist(samples[i]), 1)
        u = st.uniform.rvs(loc=0, scale=1)

        if u <= alpha:
            samples[i + 1] = xstar  # Take the candidate
            acc += 1
        else:
            samples[i + 1] = samples[i]  # Do not take the candidate

    fsamples = samples[burnIn:, :]
    fsamples = fsamples[::lag, :]

    return fsamples, acc/Nt


def mha(x0, d, targetDist, propRVS, propPDF, N, burnIn, lag):
    '''METROPOLIS-HASTINGS ALGORITHM
    Input 
        d          = dimension 
        x0         = initial sample
        targetDist = target function
        propRVS    = proposal RVS. IT CAN BE ASYMMETRICAL!!!
        propPDF    = proposal distribution (likelihood)
        N          = desired number of samples
        burnIn     = number of samples for the burnIn period
        lag        = number of samples for the lag period
    Output:
        fsamples = samples obtained from the Markov Chain
        acc/NT   = acceptance rate
    '''

    Nt = burnIn + N*lag # Total number of samples I need to generate in order 
                        # to get N samples considering burnIn and Lag periods
    samples = np.zeros((Nt, d))
    samples[0, :] = x0  # Initial seed for the chain
    acc = 0  # Count the number of accepted samples

    for i in range(Nt - 1):
        xstar = propRVS(samples[i])  # Drawing samples from a proposal PDF
        # c = correction factor of the asymmetric proposal distribution MHA
        c = propPDF(samples[i], xstar)/propPDF(xstar, samples[i])
        alpha = min(c*targetDist(xstar)/targetDist(samples[i]), 1)
        u = st.uniform.rvs(loc=0, scale=1)

        if u <= alpha:
            samples[i + 1] = xstar  # Take the candidate
            acc += 1
        else:
            samples[i + 1] = samples[i]  # Do not take the candidate

    fsamples = samples[burnIn:, :]
    fsamples = fsamples[::lag, :]

    return fsamples, acc/Nt


def gibbs():

    return None

def mma(theta0, g0, N, g_lim, b, pim, spread=1, g_failure='>=0'):
    ''' Modified Metropolis Algorithm

    Usage:
        theta, g = MMA(theta0, g0, N, g_lim, b, pim, spread=1)

    Input parameters:
        theta0: initial state of the Markov chain (seed)
        g0:     evaluation on the g_lim of theta0, i.e., g_lim(theta_0)
        N:      number of samples to be drawn
        g_lim:  limit state function g(x)
        b:      threshold (it defines the failure region F = {x : g(x) > b})
        pim:    list with the marginal PDFs of theta_1, to theta_d
        spread: spread of the proposal PDF (spread=1 by default)
        g_failure:     definition of the failure region; the possible options
                       are: >=0' (default) and '<=0'

    Output parameters:
        theta:      sampled points
        g:          g_lim(theta) for all samples
        num_eval_g: number of evaluations of the limit state function g_lim
        ar_rate:    acceptance-rejection rate
    '''
    if g_failure == '>=0':
        sign_g = 1
        if g0 < b:
            raise Exception('The initial sample does not belong to F')
    elif g_failure == '<=0':
        sign_g = -1
        if g0 < b:
            raise Exception('The initial sample does not belong to F')
    else:
        raise Exception("g_failure must be either '>=0' or '<=0'.")

    d = len(theta0)  # number of parameters (dimension of theta)

    if len(pim) != d:
        raise Exception('"pim" and "theta0" should have the same length')

    theta = np.zeros((N,d));   theta[0,:] = theta0
    g     = np.zeros(N);       g[0]       = g0
    num_eval_g = 0  # number of evaluations of g

    for i in range(N-1):
        # generate a candidate state hat_xi
        xi = np.zeros(d)

        for k in range(d):
            # the proposal PDFs are defined (the must be symmetric)
            # we will use a uniform PDF in [loc, loc+scale]
            Sk = st.uniform(loc=theta[i,k] - spread, scale=2*spread)

            # a sample is drawn from the proposal Sk
            hat_xi_k = Sk.rvs()

            # compute the acceptance ratio
            r = pim[k].pdf(hat_xi_k)/pim[k].pdf(theta[i][k])    # eq. 8

            # acceptance/rejection step:                        # eq. 9
            if np.random.rand() <= min(1, r):
                xi[k] = hat_xi_k     # accept the candidate
            else:
                xi[k] = theta[i][k]  # reject the candidate

        # check whether xi \in F by system analysis
        gg = sign_g*g_lim(xi)
        num_eval_g += 1
        if gg > b:                                              # eq. 10
            # xi belongs to the failure region
            theta[i+1, :] = xi
            g[i+1] = gg
        else:
            # xi does not belong to the failure region
            theta[i+1, :] = theta[i, :]
            g[i+1] = g[i]

    # estimation of the acceptance-rejection rate
    ar_rate = N/(num_eval_g + 1)

    # return theta, its corresponding g, the number of evaluations of g_lim,
    # and the acceptance-rejection rate
    return theta, sign_g*g, num_eval_g, ar_rate



if __name__ == "__main__":

    # ------------------------------------------------------------------------ #
    #                        Example using mha function                        #
    # ------------------------------------------------------------------------ #
    import matplotlib.pyplot as plt
    from scipy import special as spc

    def tarDist(x):
        b = 1
        y = 1.5
        if x > 0:
            return ((b**x)/spc.gamma(x))*(y**(x-1))*(np.exp(-(b*y))
                                                     *np.sin(np.pi*x)**2)
        else:
            return 0

    def proposalRVS_MHA(xi):
        std = 4
        return st.norm.rvs(loc=xi, scale=std)
    
    def proposalPDF_MHA(xi, x0):
        std = 4
        return st.norm.pdf(xi, loc=x0, scale=std)
    
    # Some needed initial values
    N = 3000  # Total number of samples
    burnin = 200   # Number of burn-in samples
    lag = 10
    x0 = 2.5  # Starting point of the chain

    # for plotting the real curve (target)
    dom = np.linspace(0, 10, 1000)
    ran = np.zeros(len(dom))
    for i in range(len(dom)):
        ran[i] = tarDist(dom[i])
    ran /= np.trapz(ran, dom)
    
    # calling mha
    samples, acceptance = mha(x0, 1, tarDist, proposalRVS_MHA, proposalPDF_MHA, 
                              10000, 1000, 5)
    
    # plotting area
    plt.plot(dom, ran, linewidth=2, color='k')
    plt.hist(samples, density=True, bins=50, edgecolor='k')
    plt.show()
    

    # print('JJSG: Hello world!')