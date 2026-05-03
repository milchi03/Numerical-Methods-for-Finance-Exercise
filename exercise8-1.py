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
    u_0 = np.maximum(np.exp(x)-1,0)
    return u_0

def buildMassBS(N, h):
    M = h/6 * ( diags([1]*(N-1),offsets=-1) + diags([4]*N) + diags([1]*(N-1),offsets=1) )
    return M

def buildABS(N, h, sigma, r):
    A1 = sigma**2/(2*h) * ( diags([-1]*(N-1),offsets=-1) + diags([2]*N) + diags([-1]*(N-1),offsets=1) )
    A2 = -(r-1/2*sigma**2)/2 * ( diags([1]*(N-1),offsets=-1) + diags([-1]*(N-1),offsets=1) )
    A3 = r*buildMassBS(N,h)

    A = A1+A2+A3

    return A


# Solver
def FEM_theta(N, M, R, B, r, sigma, K, T, theta):
    h = (R + np.log(B/K))/(N+1)
    k = T/M
    
    Mmat = buildMassBS(N, h)
    Amat = buildABS(N, h, sigma, r)

    Bmat = Mmat + k*theta*Amat
    Cmat = Mmat - k*(1-theta)*Amat

    space_grid = np.linspace(-R+h, np.log(B/K)-h, N)
    u_old = [initial_value(x) for x in space_grid]

    for m in range(1, M+1):
        u_new = spsolve(Bmat, Cmat@u_old)
        u_old = u_new
    
    u_sol = u_new
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
    uout = FEM_theta(N, M, R, B, r, sigma, K, T, theta=0.5)                                         # price of up-and-out barrier option

    # transform solution (Log-moneyness)
    uout = uout*K


    # compute up-and-in barrier
    ubs = bseucall(S, T, K, r, sigma)
    uin = ubs - uout

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