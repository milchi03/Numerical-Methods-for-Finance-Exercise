import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def initial_value(x, K):
    arg = np.exp(x)
    u_0 = np.maximum(arg - K, 0)
    return u_0

def initial_value1(u_final, K1):
    return np.maximum(u_final - K1, 0)


 
def buildMassBS(N, h):
    M = h/6 * ( diags([1]*(N-2),offsets=-1) + diags([4]*(N-1)) + diags([1]*(N-2),offsets=1) )
    return M.tocsc()

def buildABS(N, h, sigma, r):
    A1 = sigma**2/(2*h) * ( diags([-1]*(N-2),offsets=-1) + diags([2]*(N-1)) + diags([-1]*(N-2),offsets=1) )
    A2 = -(r-1/2*sigma**2)/2 * ( diags([1]*(N-2),offsets=-1) + diags([-1]*(N-2),offsets=1) )
    A3 = r*buildMassBS(N,h)

    A = A1+A2+A3

    return A.tocsc()

def FEM_theta(N, M, R, r, sigma, T, theta, u_0): #T = T1-T or just T = T
    h = 2*R/N
    k = T/M 

    Mmat = buildMassBS(N,h)
    Amat = buildABS(N, h, sigma, r)
    
    u_old = u_0

    Bmat = Mmat + k*theta*Amat
    Cmat = Mmat - k*(1-theta)*Amat
    
    for _ in range(1,M+1):
        RHS = Cmat @ u_old
        u_new = spsolve(Bmat, RHS)
        u_old = u_new

    return u_new # type: ignore

def solve_fem_two_stage(N, M, R, sigma, r, K, K1, T, T1, theta):
    #general
    k = T1/M

    #stage 1
    x = np.array([-R + i*2*R/N for i in range(1, N)])

    M1 = int(round((T1 - T) / k)) #steps in stage 1

    u_0 = initial_value(x, K) #underlying option
    u_final = FEM_theta(N, M1, R, r, sigma, T1-T, theta, u_0)

    #sage 2
    M2 = int(round(T / k)) #steps in stage 2

    v_0 = initial_value1(u_final, K1) #compound option
    v_final = FEM_theta(N, M2, R, r, sigma, T, theta, v_0)

    return [x, u_final, v_final]


# Parameters
sigma = 0.3
T = 1
T1 = 1.5
r = 0.01
K = 10
K1 = 15
R = 6
N = 3*2**8 - 1
M = 3*2**8

theta = 0.5
x, u_final, v_final = solve_fem_two_stage(N, M, R, sigma, r, K, K1, T, T1, theta)

mask = np.abs(x) <= 4
x_plot = x[mask]

plt.plot(np.exp(x_plot), v_final[mask], 'b', linewidth=2, label='Compound option value')
plt.plot(np.exp(x_plot), u_final[mask], 'r', linewidth=2, label='Initial condition/European call value')
plt.xlabel('Spot price')
plt.ylabel('Option value')
plt.title('Compound call in the BS model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()