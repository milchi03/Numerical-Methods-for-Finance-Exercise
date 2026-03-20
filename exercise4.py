import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy.linalg as lin



def alpha(x):
    return 1+x**2


def beta(x):
    return 2*x


def gamma(x):
    return np.pi**2 * x**2



def build_massMatrix(N):
    h = 1/(N+1)

    M = np.diag([2/3*h]*N) + np.diag([h/6]*(N-1), k=-1) + np.diag([h/6]*(N-1), k=1)

    return M


def build_rigidityMatrix(N, alpha, beta, gamma):
    h = 1/(N+1)
    A = np.zeros((N,N))

    for i in range(N+1):
        xi = i*h
        xi1 = xi+h
        x_mid = (xi+xi1)/2
        if i == 0:
            A[i,i] += h/6 * (
                alpha(xi)*(1/h)**2 +
                4*(alpha(x_mid)*(1/h)**2 + beta(x_mid)*(1/h)*1/2 + gamma(x_mid)*1/2*1/2) +
                alpha(xi1)*(1/h)**2 + beta(xi1)*(1/h)*1 + gamma(xi1)*1*1
                )
        elif i == N:
            A[i-1,i-1] += h/6 * (
                alpha(xi)*(-1/h)**2 + beta(xi)*(-1/h)*1 + gamma(xi)*1*1 +
                4*(alpha(x_mid)*(-1/h)**2 + beta(x_mid)*(-1/h)*1/2 + gamma(x_mid)*1/2*1/2) +
                alpha(xi1)*(-1/h)**2
                )
        elif i>0 and i<N:
            A[i-1,i-1] += h/6 * (
                alpha(xi)*(-1/h)**2 + beta(xi)*(-1/h)*1 + gamma(xi)*1*1 +
                4*(alpha(x_mid)*(1/h)**2 + beta(x_mid)*(-1/h)*1/2 + gamma(x_mid)*1/2*1/2) +
                alpha(xi1)*(-1/h)**2
                )
            A[i,i] +=h/6 * (
                alpha(xi)*(1/h)**2 +
                4*(alpha(x_mid)*(1/h)**2 + beta(x_mid)*(1/h)*1/2 + gamma(x_mid)*1/2*1/2) +
                alpha(xi1)*(1/h)**2 + beta(xi1)*(1/h)*1 + gamma(xi1)*1*1
                )
            A[i-1,i] += h/6 * (
                alpha(xi)*(-1/h)*(1/h) + beta(xi)*(1/h)*1 + 
                4*(alpha(x_mid)*(-1/h)*(1/h) + beta(x_mid)*(1/h)*1/2 + gamma(x_mid)*1/2*1/2) +
                alpha(xi1)*(-1/h)*(1/h)
            )
            A[i,i-1] += h/6 * (
                alpha(xi)*(-1/h)*(1/h) +
                4*(alpha(x_mid)*(-1/h)*(1/h) + beta(x_mid)*(-1/h)*1/2 + gamma(x_mid)*1/2*1/2) +
                alpha(xi1)*(-1/h)*(1/h) + beta(xi1)*(-1/h)*1
            )
    return A


def f(t, x):
    def u(t,x):
        return np.exp(-t)*np.sin(np.pi*x)
    
    return u(t,x)*(np.pi**2-1) + 2*x**2*np.pi**2*u(t,x)


def initial_value(x):
    return np.sin(np.pi*x)


def exact_solution_at_1(x):
    return np.exp(-1)*np.sin(np.pi*x)


def build_F(t, N):
    h = 1 / (N + 1)
    X = np.linspace(0, 1, N + 2)[1:N+1]

    X_mid_left = X-h/2
    X_mid_right = X+h/2
    F = (h/3)*(f(t,X_mid_left) + f(t,X) +f(t,X_mid_right))
    return F


def FEM_theta(N, M, theta):
    Mmat = build_massMatrix(N)
    A = build_rigidityMatrix(N, alpha, beta, gamma)
    

    k = 1/M
    X = np.linspace(0, 1, N+2)[1:N+1]
    T = np.linspace(0, 1, M+1)
    u = np.zeros((N,M+1))
    f = np.zeros((N,M+1))
    F = np.zeros((N,N))

    for j, t in enumerate(T):
        f[:,j] = build_F(t,N)

    u[:,0] = initial_value(X)
    
    LHS = Mmat + k*theta*A
    LHS = sp.csr_matrix(LHS)
    
    for i in range(1,M+1):
        F_vec = theta*f[:,i] + (1-theta)*f[:,i-1]

        RHS = (Mmat-k*(1-theta)*A) @ u[:,i-1] + k*F_vec
        u[:,i] = spsolve(LHS, RHS)

    return u[:,-1]

#print(FEM_theta(3,3,0.5))


#### error analysis ####
#the code runs significantly longer. I haven't stoped the time but 10-15 minutes. For the M=4**l case the error is O(n) for theta=0.5. For theta=0.3 it doesn't converge (as always and with similar arguments). For theta=1 the error still falls linearly.
#For M=2**l, theta =0.5 the error falls quadratically. For theta=1 linearly. For theta=0.3 it doesm't converge.
#I saved the theta=0.5, M=4**l version as a png and attatched it to the solution.

nb_samples = 5
N = np.array([2**l-1 for l in range(2,nb_samples+2)])
M = np.array([4**l for l in range(2,nb_samples+2)]) # M = np.array([2**l for l in range(6,nb_samples+1)])
theta = 0.3

#### Do not change any code below! ####
l2error = np.zeros(nb_samples)
k = 1 / M


try:
    for i in range(nb_samples):
        l2error[i] = (1 / (N[i] + 1)) ** (1 / 2) * lin.norm(
            exact_solution_at_1((1 / (N[i] + 1)) * (np.arange(N[i]) + 1))
            - FEM_theta(N[i], M[i], theta),
            ord=2,
        )
        if np.isnan(l2error[i]) == True:
            raise Exception("Error unbounded. Plots not shown.")
    conv_rate = np.polyfit(np.log(k), np.log(l2error), deg=1)
    if not conv_rate[0] >= 0:
        raise Exception("Error unbounded. Plots not shown.")
    print(
        f"FEM with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}"
    )
    plt.figure(figsize=[10, 6])
    plt.loglog(k, l2error, "-x", label="error")
    plt.loglog(k, k, "--", label="$O(k)$")
    plt.loglog(k, k**2, "--", label="$O(k^2)$")
    plt.title("$L^2$ convergence rate", fontsize=13)
    plt.xlabel("$k$", fontsize=13)
    plt.ylabel("error", fontsize=13)
    plt.legend()
    plt.plot()
    plt.show()
except Exception as e:
    print(e)
