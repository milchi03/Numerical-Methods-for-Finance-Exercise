import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import matplotlib.pyplot as plt

def Build_massMatrix(N, R):
    """
    Constructs and outputs the mass matrix M.
    """
    # TODO: Implement this function
    pass

def Build_BSMatrix(N, sigma, r, R):
    """
    Constructs and outputs the stiffness matrix A^{BS}.
    """
    # TODO: Implement this function
    pass

def u0(x, K0, K2):
    """
    Outputs the initial vector \underline{u}^0.
    """
    # TODO: Implement this function
    pass

def bs_formula_C(s, t, sigma, K, r):
    """
    Exact Black-Scholes formula for a European Call Option.
    (Realized for you as per the exam hint).
    """
    if t == 0:
        return np.maximum(s - K, 0)
    
    d1 = (np.log(s / K) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

def bs_formula_BF(s, t, sigma, K0, K2, r):
    """
    Outputs the exact butterfly option price.
    """
    # TODO: Implement this function (Hint: use bs_formula_C)
    pass

def FEM_theta(N, M_steps, R, sigma, r, K0, K2, T, theta):
    """
    Time-stepping theta scheme solver.
    """
    k = T / M_steps
    x_line = np.linspace(-R, R, N + 2)[1:-1]
    
    # Get initial condition
    u_sol = u0(x_line, K0, K2)
    
    # Get matrices
    M_mat = Build_massMatrix(N, R)
    A_mat = Build_BSMatrix(N, sigma, r, R)
    
    # Setup step matrices
    B_mat = M_mat + k * theta * A_mat
    C_mat = M_mat - k * (1 - theta) * A_mat
    
    for _ in range(M_steps):
        RHS = C_mat @ u_sol
        u_sol = spsolve(B_mat, RHS)
        
    return u_sol


if __name__ == "__main__":
    #### Parameter setting
    # TODO: Set the parameters according to task f
    R = None 
    T = None
    K0 = None
    K2 = None
    sigma = None
    r = None
    theta = None
    
    # TODO: define the arrays for l, N, and M as requested
    l_vals = []  
    N_vals = [] 
    M_vals = [] 

    ############################ Do not change any code below! ############################
    if l_vals:  # Will only run once you fill in the parameters
        error = np.zeros(len(l_vals))
        k_vals = T / np.array(M_vals)
        
        try:
            # Convergence rate in L^inf(G)
            for i, (N, M_steps) in enumerate(zip(N_vals, M_vals)):
                grid = np.linspace(-R, R, N + 2)[1:-1]
                u_num = FEM_theta(N, M_steps, R, sigma, r, K0, K2, T, theta)
                u_exact = bs_formula_BF(np.exp(grid), T, sigma, K0, K2, r)
                
                err = u_exact - u_num
                error[i] = np.max(np.abs(err)) 
                
                if np.isnan(error[i]):
                    raise Exception(f"Error unbounded at l={l_vals[i]}.")
                    
            conv_rate = np.polyfit(np.log(k_vals), np.log(error), deg=1)
            if conv_rate[0] < 0:
                raise Exception("Error did not converge. Plots not shown.")
                
            print(f"Convergence rate in L^inf norm with respect to time step k: {conv_rate[0]}")
            # TODO: Answer the question - does it coincide with the theoretical convergence rate?
            
            plt.figure(figsize=[10, 6])
            plt.loglog(k_vals, error, "-x", label="$L^\infty$ error")
            plt.loglog(k_vals, k_vals, "--", label="$O(k)$")
            plt.loglog(k_vals, k_vals**2, "--", label="$O(k^2)$")
            plt.title("Convergence rate", fontsize=13)
            plt.xlabel("$k$", fontsize=13)
            plt.ylabel("error", fontsize=13)
            plt.legend()
            plt.show()

            # Plot for l=8 around x=R
            if 8 in l_vals:
                l_index = l_vals.index(8)
                N8 = N_vals[l_index]
                M8 = M_vals[l_index]
                grid8 = np.linspace(-R, R, N8 + 2)[1:-1]
                u_num8 = FEM_theta(N8, M8, R, sigma, r, K0, K2, T, theta)
                u_exact8 = bs_formula_BF(np.exp(grid8), T, sigma, K0, K2, r)
                
                plt.figure(figsize=[10, 6])
                plt.plot(grid8, u_exact8, label='Exact Analytical Solution', alpha=0.8)
                plt.plot(grid8, u_num8, label='Numerical FEM Solution', linestyle='--')
                plt.xlim([R - 1.5, R]) 
                plt.title("Boundary Behavior at $x=R$ ($l=8$)", fontsize=13)
                plt.xlabel("Log Price ($x$)", fontsize=13)
                plt.ylabel("Option Price ($V$)", fontsize=13)
                plt.legend()
                plt.show()
                
                # TODO: Answer the question - What do you observe around x=R? 
                # Would you expect the same observation for a European call option?
            
        except Exception as e:
            print(e)