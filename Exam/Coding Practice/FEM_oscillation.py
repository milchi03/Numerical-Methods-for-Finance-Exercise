import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import numpy.linalg as lin


def kappa_integral(x,y):
    return 0.5*(y**2-x**2) + y-x


def build_massMatrix(N):
    a = 1/6 * np.ones(N-1)
    M = sp.diags(a,-1) + sp.diags(a,1) + 2/3 * sp.eye(N)
    return 1/(N+1) * M


def build_rigidityMatrix(N):
    M = np.zeros((N,N))
    
    for i in range(N):
        M[i,i] = (N+1)**2 * (kappa_integral((i)/(N+1), (i+1)/(N+1)) + kappa_integral((i+1)/(N+1), (i+2)/(N+1)))

    for i in range(1,N):
        M[i, i-1] = -(N+1)**2 * kappa_integral((i)/(N+1), (i+1)/(N+1))

    for i in range(N-1):
        M[i, i+1] = -(N+1)**2 * kappa_integral((i+1)/(N+1), (i+2)/(N+1))

    return sp.csr_matrix(M)


def f(t, x):
    return x * t


def initial_value(x):
    return ((x >= 0.25) * (x <= 0.75)).astype(float)



def build_F(t, N):
    X = np.linspace(0, 1, N + 2)
    h = np.diff(X)
    a = f(t, X[1:-1])
    ab2 = f(t, (X[:-2] + X[1:-1]) / 2)
    b = f(t, (X[1:-1] + X[2:]) / 2)

    return a * (h[:-1] + h[1:]) / 6 + ab2 * h[:-1] / 3 + b * h[1:] / 3


def FEM_theta(N, M, theta, beta=1):
    h = 1/(N+1)
    # k = 1/M

    x = np.array([h*i for i in range(1, N+1)][0:N+1])
    #print(f'x = {x}, should have lengh ={N}')

    M_mat = build_massMatrix(N)
    A_mat = build_rigidityMatrix(N)
    
    u_est = initial_value(x)

    for j in range(M):
        t_beta = (j/M)**beta
        t1_beta = ((j+1)/M)**beta

        k = t1_beta - t_beta

        B = M_mat + k*theta*A_mat
        C = M_mat - k*(1-theta)*A_mat

        F_m = build_F(t_beta, N)
        F_m1 = build_F(t1_beta, N)
        F_vec = k*theta*F_m1 + k*(1-theta)*F_m

        RHS = C @ u_est + F_vec

        u_est = spsolve(B, RHS)
    return u_est

print(FEM_theta(9,9, 0.5, 17))

# #### error analysis ####
nb_samples = 3
N = np.power(2, np.arange(9, 9 + nb_samples)) - 1
M = np.power(2, np.arange(9, 9 + nb_samples))
theta = 0.5
beta = 1 # set beta according to b) and d)

# d)
# We cannot expect O(h**2 + k**2) convergence because we have a step function. 
# True step functions are not in H2 because to be in H2 one needs to be continuous (result from exercise).

# conv_rate = # Estimate the convergence rate

# print(
#     f"FEM with theta={theta}, beta={beta}: Convergence rate in discrete l^2 norm with respect to time step $k$: {conv_rate}"
# )