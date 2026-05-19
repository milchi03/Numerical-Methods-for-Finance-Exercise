import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def kappa_integral(x,y):
    
    upper = 1/2*y**2 + y
    lower = 1/2*x**2 + x

    return upper - lower

def simpson_rule(f, a, b):
    fact = (b-a)/6
    mid = 4*f((a+b)/2)

    return fact*(f(a) + mid + f(b))

def build_massMatrix(N):
    h = 1/(N+1)
    M_mat = 1/6*h * (np.diag([4]*N) + np.diag([1]*(N-1), k=-1) + np.diag([1]*(N-1), k=1))
    M_mat = sp.csc_matrix(M_mat)
    return M_mat

def build_rigidityMatrix(N):
#     # todo 3 b)
#     # Be careful with the indices!
#     # kappa_integral could be helpful here
    
    def del_phi_del_phi(i, j, N):
        h = 1/(N+1)

        if i==j: return 2*(1/h)**2
        elif np.abs(i-j) == 1: return -(1/h)**2
        else: return 0

    A_mat = np.zeros((N,N))
    h = 1/(N+1)

#     # The case of i=j
    for i in range(N): #i = 1, ..., N
        A_mat[i,i] = del_phi_del_phi(i, i, N) * kappa_integral(i*h, (i+2)*h)

#     # The case of i-j=1
    for i in range(1,N):
        A_mat[i, i-1] = del_phi_del_phi(i, i-1, N) * kappa_integral(i*h, (i+1)*h)

#     # The case of j-i=1
    for i in range(N-1):
        A_mat[i, i+1] = del_phi_del_phi(i, i+1, N) * kappa_integral((i+1)*h, (i+2)*h)

    return A_mat

def f(t,x):
    return (-1 +(x+1) * np.pi**2) * np.exp**(-t) * np.sin(np.pi*x) - np.pi* np.exp**(-t) * np.cos(np.pi*x)

# u(t,x) = exp**(-t) * sin(pi*x)
# del_t u = - exp**(-t) * sin(pi*x)
# del_x u = exp**(-t) * cos(pi*x) * pi
# I := kappa * del_x u = exp**(-t) * cos(pi*x) * pi * (x+1)
# del_x (I) = exp**(-t) * (-sin(pi*x)) * pi**2 * (x+1) + exp**(-t) * cos(pi*x) * pi
# =  -(x+1) * pi**2 * exp**(-t) * sin(pi*x) + pi* exp**(-t) * cos(pi*x)

# del_t u - del_xx u = - exp**(-t) * sin(pi*x)  + (x+1) * pi**2 * exp**(-t) * sin(pi*x) - pi* exp**(-t) * cos(pi*x)
# = (-1 +(x+1) * pi**2) * exp**(-t) * sin(pi*x) - pi* exp**(-t) * cos(pi*x) = f(t,x) => check

# u_0(x) = sin(pi*x) = u(0,x) = exp**(0) * sin(pi*x) => check

# u(t,0) = exp**(-t) * sin(0) = 0 
# u(t,1) = exp**(-t) * sin(pi) = 0 => check

def initial_value(x):
    u0 = np.sin(np.pi*x)
    return u0

def exact_solution_at_1(x):
    
    return


# def build_F(t,N):
#     # todo 3 d)
#     return


# def FEM_theta(N,M,theta):
#     # todo 3 e)
#     return


# #### error analysis ####
# nb_samples = 5
# N = # fill in this line for f)-g)
# M = # fill in this line for f)-g)
# theta= # fill in this line for f)-g)


# #### Do not change any code below! ####
# l2error = np.zeros(nb_samples) 
# k =  1 / M

# try:
#    for i in range(nb_samples):
#       l2error[i] = (1 / (N[i]+1)) ** (1 / 2) * lin.norm(exact_solution_at_1((1/(N[i]+1))*(np.arange(N[i])+1)) - FEM_theta(N[i], M[i],theta), ord=2)
#       if np.isnan(l2error[i])==True:
#           raise Exception("Error unbounded. Plots not shown.")
#    conv_rate = np.polyfit(np.log(k), np.log(l2error), deg=1)
#    if conv_rate[0]<0:
#        raise Exception("Error unbounded. Plots not shown.")
#    print(f"FEM method with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}")
#    plt.figure(figsize=[10, 6])
#    plt.loglog(k, l2error, '-x', label='error')
#    plt.loglog(k, k, '--', label='$O(k)$')
#    plt.loglog(k, k**2, '--', label='$O(k^2)$')
#    plt.title('$L^2$ convergence rate', fontsize=13)
#    plt.xlabel('$k$', fontsize=13)
#    plt.ylabel('error', fontsize=13)
#    plt.legend()
#    plt.plot()
#    plt.show()
# except Exception as e:
#     print(e)