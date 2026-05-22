from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import numpy.linalg as lin
from scipy.stats import norm
import matplotlib.pyplot as plt

def sigma(t):
    return 0.3

def sigmaT(t, T):
    return sigma(T-t)

def delta(t):
    return 0.02*t

def deltaT(t, T):
    return delta(T-t)

def r(t):
    return 0.05

def rT(t, T):
    return r(T-t)


def buildMassBS(N, R):
    """
    Parameters:
    - N (int)
    - R (float)

    Returns:
    - MassBS : The (N x N) mass matrix.
    """
    h = 2*R/(N+1)

    MassBS = h/6 * (np.diag([4]*N) + np.diag([1]*(N-1), k=-1) + np.diag([1]*(N-1), k=1))
    MassBS = sp.csr_matrix(MassBS)

    return MassBS


def buildABS(N, R, sigmaT, rT, deltaT, t, T):
    """
    Parameters:
    - N (int)
    - R (float)
    - sigmaT (function)
    - rT (function)
    - deltaT (function)
    - t (float)
    - T (float)

    Returns:
    - ABS : The (N x N) stiffness matrix.
    """
    h = 2*R/(N+1)

    A1 = sigmaT(t, T)**2/(2*h) * (np.diag([2]*N) + np.diag([-1]*(N-1), k=-1) + np.diag([-1]*(N-1), k=1))
    A1 = sp.csr_matrix(A1)

    A2 = (sigmaT(t, T)**2/2 + deltaT(t, T) - rT(t, T)) * (np.diag([0]*N) + np.diag([-1/2]*(N-1), k=-1) + np.diag([1/2]*(N-1), k=1))
    A2 = sp.csr_matrix(A2)

    A3 = rT(t, T) * buildMassBS(N, R)

    return A1 + A2 + A3


def u_init(x, K):
    return np.maximum(0, K - np.exp(x))


def exactu(x):
    d2 = (x - 0.005) / 0.3
    d1 = d2 + 0.3
    return np.exp(-0.05) * norm.cdf(-d2) - np.exp(x - 0.01) * norm.cdf(-d1)


def FEM_theta(N, M, R, sigmaT, rT, deltaT, K, T, theta):
    """
    Parameters:
    - N (int)
    - M (int)
    - R (float)
    - sigmaT (function)
    - rT (function)
    - deltaT (function)
    - K (float)
    - T (float)
    - theta (float)

    Returns:
    - u_sol (numpy.ndarray): The solution vector on the grid x_i, i=1,2,...,N.
    """
    h = 2*R/(N+1)
    k = T/M

    t_line = np.array([j*k for j in range(M)])
    #print(t_line)
    
    x_line = np.array([-R + i*h for i in range(1, N+1)])
    #print(x_line)

    u_sol = u_init(x_line, K)
    #print(u_sol)
    
    for t in t_line:
        M_mat = buildMassBS(N, R)
        A_mat = buildABS(N, R, sigmaT, rT, deltaT, t, T)

        B_mat = M_mat + k*theta*A_mat
        C_mat = M_mat - k*(1-theta)*A_mat

        RHS = C_mat @ u_sol
        u_sol = spsolve(B_mat, RHS)
    return u_sol


if __name__ == "__main__":


    #### Parameter setting
    T = 1
    R = 4
    K = 1
    theta = 0.5
    N = np.array([2**l-1 for l in range(4,9)])
    M = np.array([2**l for l in range(4,9)])
    #print(FEM_theta(2**4-1, 2**4, R, sigmaT, rT, deltaT, K, T, theta))

    # comments:
    # FEM method with theta=0.5 converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: 2.081816509373512
    # This is to be expected as FEM with u 3-time cont. differentiable wrt time and theta = 1/2 the L2-error converges unconditionally 
    # in O(h**2 + k**2) order. As in our case h \in =(k**2) we get the convergence as in the plot.
    # Also note: for x -> R the localization error grows exponentially in |x|. As stated in the hint the plot looks at |x| < R/2
    # where for sufficiently large R (which we have as our error is tiny) the error is small.
    # This is why we chose G~ = (-R/2, R/2) instead of G = (-R, R).

    ############################ Do not change any code below! ############################
    G = R / 2
    error = np.zeros(5)
    k = T / M

    
    try:
        for i in range(5):
            grid = np.linspace(-R, R, N[i] + 2)[1:-1]
            ind = (np.abs(grid)) < G
            err = exactu(grid) - FEM_theta(N[i], M[i], R, sigmaT, rT, deltaT, K, T, theta)
            error[i] = (1 / (N[i] + 1)) ** (1 / 2) * lin.norm(err[ind], ord=2)
            if np.isnan(error[i]) == True:
                raise Exception("Error unbounded. Plots not shown.")
        conv_rate = np.polyfit(np.log(k), np.log(error), deg=1)
        if conv_rate[0] < 0:
            raise Exception("Error did not converge. Plots not shown.")
        print(
            f"FEM method with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}"
        )
        plt.figure(figsize=[10, 6]) #type: ignore
        plt.loglog(k, error, "-x", label="error")
        plt.loglog(k, k, "--", label="$O(k)$")
        plt.loglog(k, k**2, "--", label="$O(k^2)$")
        plt.title("Convergence rate", fontsize=13)
        plt.xlabel("$k$", fontsize=13)
        plt.ylabel("error", fontsize=13)
        plt.legend()
        plt.plot()
        #plt.savefig(Path.home() / "questions" / "Problem2" / "plot_BS.pdf", format="pdf")
        plt.show()
    except Exception as e:
        print(e)
