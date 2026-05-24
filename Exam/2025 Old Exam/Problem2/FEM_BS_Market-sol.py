import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import matplotlib.pyplot as plt

def Build_massMatrix(N, R):
    """
    Constructs and outputs the mass matrix M.
    """
    h = 2*R/(N+1)
    
    M_mat = h/6 * (np.diag([4]*N) + np.diag([1]*(N-1), k=1) + np.diag([1]*(N-1), k=-1))
    
    M_mat = sp.csr_matrix(M_mat)
    return M_mat

#print(Build_massMatrix(2,9)) #BIBACCTO

def Build_BSMatrix(N, sigma, r, R):
    """
    Constructs and outputs the stiffness matrix A^{BS}.
    """
    h = 2*R/(N+1)

    A1 = sigma**2/(2*h) * (np.diag([2]*N) + np.diag([-1]*(N-1), k=1) + np.diag([-1]*(N-1), k=-1))
    A2 = (sigma**2/2 - r) * (np.diag([0]*N) + np.diag([+1/2]*(N-1), k=1) + np.diag([-1/2]*(N-1), k=-1))
    A3 = r * Build_massMatrix(N, R)

    A_mat = sp.csr_matrix(A1) + sp.csr_matrix(A2) + A3
    return A_mat

#print(Build_BSMatrix(2, 1, 3, 3)) #BIBACCTO

def u0(x, K0, K2):

    K1 = (K0 + K2)/2

    g0 = np.maximum(0, np.exp(x) - K0)
    g1 = np.maximum(0, np.exp(x) - K1)
    g2 = np.maximum(0, np.exp(x) - K2)

    return g0 - 2*g1 + g2 
    
# plt.plot(np.linspace(-0.8,1.2,100), u0(np.linspace(-0.8,1.2,100), 0.9, 1.1))
# plt.show()

def bs_formula_C(s, t, sigma, K, r):
    """
    Exact Black-Scholes formula for a European Call Option. (hint)
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
    K1 = (K0 + K2)/2

    C0 = bs_formula_C(s, t, sigma, K0, r)
    C1 = bs_formula_C(s, t, sigma, K1, r)
    C2 = bs_formula_C(s, t, sigma, K2, r)
    return C0 - 2*C1 + C2

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
    R = 3
    T = 2
    K0 = 0.5
    K2 = 1.5
    sigma = 0.4
    r = 0.06
    theta = 0.5
    
    l_vals = [l for l in range(6,11)]  
    N_vals = [2**l - 1 for l in l_vals] 
    M_vals = [2**l for l in l_vals]

    # comments:
    # The convergence rate is approximately O(h**2 + k**2) (which in our case is O(h**2)). The empricial result is: 2.0788878140849003.
    # Theory guarantees FEM finds for u \in H2(G) with theta = 1/2 the solution in O(h**2).
    # u \in H1(G) is clear bc we showed in the exercise that any continuous function that has only finitely-many non-differentiable points are weakly differentiable.
    # This clearly fits with u0 but is clearly violated for the weak derivative of u0. u0 \in H2(G) required the weak deriviative of u0 to be in H1(G) which is not case.
    # A step function, which the weak deriviate is, is not continuous and therefore cannot be in H1(G) and the weak derivitive of u0 is a step function.
    # We still see a convergence of O(h**2) due to the smoothing in time form the Black Scholes formula.
    # Approaching R we see that the function smoothly slides into being 0 for x->R. This is because the payoff-function itself is 0 outside of (K0, K2).
    # This is different then in the standard call-option case as there we jump from a non-zero payoff to a zero-payoff in a discountinuous way once we breach the barrier.

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
            
            plt.figure(figsize=[10, 6]) #type: ignore
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
                
                plt.figure(figsize=[10, 6]) #type: ignore
                plt.plot(grid8, u_exact8, label='Exact Analytical Solution', alpha=0.8)
                plt.plot(grid8, u_num8, label='Numerical FEM Solution', linestyle='--') #type: ignore
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