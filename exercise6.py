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
    X = np.linspace(0,1,N+2)
    h = np.diff(X)
    t = [(j/M)**beta for j in range(M+1)]

    u0 = initial_value(X[1:-1])
    M_mat = build_massMatrix(N)
    A_mat = build_rigidityMatrix(N)
    F_vec = [0,0]
    F_vec[1] = build_F(t[0], N)

    for j in range(M):
        k = t[j+1]-t[j]
        B = sp.csr_matrix(M_mat + k*theta*A_mat)
        C = sp.csr_matrix(M_mat - k*(1-theta)*A_mat)

        F_vec[0] = F_vec[1] #=build_F(t[j],N)
        F_vec[1] = build_F(t[j+1],N)

        RHS = C @ u0 + k * ((1 - theta) * F_vec[0] + theta * F_vec[1])
        u0 = spsolve(B, RHS)

    return u0

def h_norm(u0, u1, N):
    h = 1/(N+1)
    S = sum([(u0[j] - u1[2*j+1])**2 for j in range(N)])
    return np.sqrt(h*S)

def conv_est(u0, u1, u2):
    return np.log(h_norm(u0, u1, N[0])/h_norm(u1, u2, N[1])) / np.log(2)

#### error analysis ####
nb_samples = 3
N = np.power(2, np.arange(9, 9 + nb_samples)) - 1
M = np.power(2, np.arange(9, 9 + nb_samples))
theta = 0.5
beta = 1 # set beta according to b) and d)

u0 = FEM_theta(N[0],M[0], theta)
u1 = FEM_theta(N[1],M[1], theta)
u2 = FEM_theta(N[2],M[2], theta)

conv_rate = conv_est(u0,u1,u2) # Estimate the convergence rate

print(
    f"FEM with theta={theta}, beta={beta}: Convergence rate in discrete l^2 norm with respect to time step $k$: {conv_rate}"
)

#test
#For beta=17, we get roughly p=2, for beta=1 we get roughly p=0.5.
#This comes from the discountinuity of the option price as t->T (or tau-> 0) violating the assumption of continuity of the solution.
#With beta increasing, we put x**beta more points to the closer maturing areas in the grid, refining the grid
#where the option becomes discontinuous so we can capture the transition from t!=0 to t=0 better.