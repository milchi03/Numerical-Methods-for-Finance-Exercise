import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

def bseucall(S, T, K, r, sigma):
    """
    Computes the value of a European put option with the Black-Scholes model using analytic formulas.

    Parameters:
        S (float or array-like): Stock prices at time 0.
        T (float): Maturity.
        K (float): Strike.
        r (float): Interest rate.
        sigma (float): Volatility.

    Returns:
        P (float or array-like): Option price at time 0.
    """
    # call option
    d1 = (np.log(S / K) + (0.5 * sigma ** 2 + r) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    P = S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
    return P

# Compute Generator/ Source Term / Initial Data
def initial_value(x):
    return u_0

def buildMassBS(N, h):
    return M

def buildABS(N, h, sigma, r):

    return A


# Solver
def FEM_theta(N, M, R, B, r, sigma, K, T, theta):
    
    return u_sol

if __name__ == '__main__':
    # Set Parameters
    N = 255                     # number of nodes
    M = 256                     # number of time steps
    R = 3                       # localization
    T = 1                       # maturity
    K = 60                      # strike
    r = 0.01                    # interest rate
    sigma = 0.3                 # volatility
    B = 80                      # barrier

    x = np.linspace(-R, np.log(B/K), N + 2)[1:-1]   # uniform grid in log-price
    S = np.exp(x)*K                                 # corresponding real prices

    # compute up-and-out barrier
    uout =                                          # price of up-and-out barrier option

    # transform solution (Log-moneyness)
    uout = 


    # compute up-and-in barrier
    ubs = bseucall(S, T, K, r, sigma)
    uin = 

    # Postprocessing
    # area of interest
    I = np.abs(x) < 0.75


    # plot solution
    plt.figure(1)
    plt.plot(S[I], uout[I], 'bx-', label='Knock-out barrier', markersize=2, linewidth=0.5)
    plt.plot(S[I], uin[I], 'go-', label='Knock-in barrier', markersize=2, linewidth=0.5, fillstyle='none')
    plt.plot(S[I], ubs[I], 'rs-', label='Plain vanilla', markersize=2, linewidth=0.5, fillstyle='none')
    plt.plot(K * np.exp(np.linspace(-0.75, np.log(B/K), 1000)), K * initial_value(np.linspace(-0.75, np.log(B/K), 1000)), 'k-', label='Payoff', linewidth=0.5)
    plt.xlabel('s')
    plt.ylabel('Option price')
    plt.legend(loc='upper right')
    plt.savefig('price.eps', format='eps')
    plt.show()