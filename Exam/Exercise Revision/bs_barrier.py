import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
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
    u_0 = np.maximum(0, np.exp(x) - 1)
    return u_0

def buildMassBS(N, h):
    fact = h/6

    s1 = np.diag([4]*N)
    s2 = np.diag([1]*(N-1), k=1)
    s3 = np.diag([1]*(N-1), k=-1)

    M = fact * (s1 + s2 + s3)
    
    M = sp.csr_matrix(M)
    
    return M

def buildABS(N, h, sigma, r):
    fact1 = sigma**2/2 / h
    fact2 = -(r - sigma**2/2)
    fact3 = r

    s11 = np.diag([2]*N)
    s12 = np.diag([-1]*(N-1), k=1)
    s13 = np.diag([-1]*(N-1), k=-1)

    s1 = fact1 * (s11 + s12 + s13)
    s1 = sp.csc_matrix(s1)

    s21 = np.diag([0]*N)
    s22 = np.diag([1/2]*(N-1), k=1)
    s23 = np.diag([-1/2]*(N-1), k=-1)

    s2 = fact2 * (s21 + s22 + s23)
    s2 = sp.csc_matrix(s2)

    s3 = fact3 * buildMassBS(N, h)

    A = s1 + s2 + s3

    return A


# Solver
def FEM_theta(N, M, R, B, r, sigma, K, T, theta):
    h = (R + np.log(B/K)) / (N+1)
    k = T/M

    M_mat = buildMassBS(N, h)
    A_mat = buildABS(N, h, sigma, r)

    B_mat = M_mat + k*theta*A_mat
    C_mat = M_mat - k*(1-theta)*A_mat

    x = np.array([-R + i*h for i in range(1, N+1)])

    u_sol = initial_value(x)
    
    for _ in range(M):
        RHS = C_mat @ u_sol
        u_sol = spsolve(B_mat, RHS)
        
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
    uout = FEM_theta(N, M, R, B, r, sigma, K, T, theta = 0.5)                                        # price of up-and-out barrier option

    # transform solution (Log-moneyness)
    uout = uout * K

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