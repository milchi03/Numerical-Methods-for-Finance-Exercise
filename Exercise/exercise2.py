import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


#K(x,y) = (1/2 * x**2 + x |x = x,y) = 1/2 * y**2 + y - 1/2 * x**2 - x
def kappa_integral(x,y):
    return 1/2 * y**2 + y - 1/2 * x**2 - x


def build_massMatrix(N):
    h = 1 / (N + 1)
    main_diag = np.full(N, 2 * h/3)
    off_diag = np.full(N-1, h/6)
    return np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

#print(build_massMatrix(3))

def build_rigidityMatrix(N):
    # todo 3 b)
    # Be careful with the indices!
    # kappa_integral could be helpful here
    # phi'_i(x) = 1/h on (i-1,i) and -1/h on (i,i+1)
    h = 1/(N+1)
    A = np.zeros((N,N))
    
    x = np.linspace(0, 1, N+2)

    # The case of i=j
    for i in range(N): #Int(kappa*(1/h)**2) = (1/h)**2 * kappa_integral
        x_id = i+1
        A[i,i] = (1/h)**2 * (kappa_integral(x[x_id-1], x[x_id]) + kappa_integral(x[x_id], x[x_id+1]))

    # The case of i-j=1
    for i in range(1,N): #Int(kappa*(1/h)*(-1/h)) = -(1/h)**2 * kappa_integral
        x_id = i+1
        A[i, i-1] = -(1/h)**2 * (kappa_integral(x[x_id-1], x[x_id]))

    # The case of j-i=1
    for i in range(N-1):
        x_id = i+1
        A[i, i+1] = -(1/h)**2 * (kappa_integral(x[x_id], x[x_id+1]))

    return A

#print(build_rigidityMatrix(3))

def f(t,x):
    return ((x+1)*np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi*x) - np.pi * np.exp(-t) * np.cos(np.pi*x)

def initial_value(x):
    return np.array([np.sin(np.pi*x_i) for x_i in x])


def exact_solution_at_1(x):
    return np.array([np.exp(-1) * np.sin(np.pi*x_i) for x_i in x])


def build_F(t,N):
    h = 1/(N+1)
    x = np.linspace(0,1, N+2)

    F_all = (h / 3) * (f(t, x - h/2) + f(t, x) + f(t, x + h/2))

    return F_all[1:-1]

#print(build_F(1,3))

def FEM_theta(N, M, theta):
    Mmat = build_massMatrix(N)
    Amat = build_rigidityMatrix(N)
    h = 1 / (N + 1)
    k = 1 / M       
    t_line = np.linspace(0, 1, M + 1)
    x = np.linspace(h, 1-h, N)

    u = initial_value(x)

    LHS = Mmat + k * theta * Amat
    RHS_matrix = Mmat - k * (1 - theta) * Amat

    for m in range(M):
        f_now = build_F(t_line[m], N)
        f_next = build_F(t_line[m+1], N)
        
        b = RHS_matrix @ u + k * (theta * f_next + (1 - theta) * f_now)
        
        u = lin.solve(LHS, b)
        
    return u

print(FEM_theta(3,9,0.5))


#### error analysis ####
# only for theta = 0.5 and 1 we get a convergent solution. As with the FDM. For theta == 1 the error term depreciates lineraly, while for theta == 0.5 it depreciates quadratically. Especially easy to see for nb_samples == 7.
# when M is written with the 4, instead of 2 the convergence happens again only at theta == 0.5 and 1. But this time the convergence rate is linear, independently of theta. This is because this time the spatial error is dominant.
nb_samples = 5
N = np.array([2**l-1 for l in range(2,nb_samples+2)])
M = np.array([4**l for l in range(2,nb_samples+2)]) # M = np.array([2**l for l in range(2,nb_samples+2)])


#### Do not change any code below! ####
l2error = np.zeros(nb_samples) 
k =  np.array([1 / m for m in M])

try:
   for i in range(nb_samples):
      l2error[i] = (1 / (N[i]+1)) ** (1 / 2) * lin.norm(exact_solution_at_1((1/(N[i]+1))*(np.arange(N[i])+1)) - FEM_theta(N[i], M[i],theta), ord=2)
      if np.isnan(l2error[i])==True:
          raise Exception("Error unbounded. Plots not shown.")
   conv_rate = np.polyfit(np.log(k), np.log(l2error), deg=1)
   if conv_rate[0]<0:
       raise Exception("Error unbounded. Plots not shown.")
   print(f"FEM method with theta={theta} converges: Convergence rate in discrete $L^2$ norm with respect to time step $k$: {conv_rate[0]}")
   plt.figure(figsize=[10, 6])
   plt.loglog(k, l2error, '-x', label='error')
   plt.loglog(k, k, '--', label='$O(k)$')
   plt.loglog(k, k**2, '--', label='$O(k^2)$')
   plt.title('$L^2$ convergence rate', fontsize=13)
   plt.xlabel('$k$', fontsize=13)
   plt.ylabel('error', fontsize=13)
   plt.legend()
   plt.plot()
   plt.show()
except Exception as e:
    print(e)