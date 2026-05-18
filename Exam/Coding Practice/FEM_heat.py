import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def kappa_integral(x,y):
    # todo 3 a)
    return 


def build_massMatrix(N):
    # todo 3 b)
    return


def build_rigidityMatrix(N):
    # todo 3 b)
    # Be careful with the indices!
    # kappa_integral could be helpful here
    
    M = np.zeros((N,N))
    
    # The case of i=j
    for i in range(N):
        M[i,i] = # to be filled

    # The case of i-j=1
    for i in range(1,N):
        M[i, i-1] = # to be filled

    # The case of j-i=1
    for i in range(N-1):
        M[i, i+1] = # to be filled

    return


def f(t,x):
    # todo 3 c)
    return
    

def initial_value(x):
    # todo 3 c)
    return


def exact_solution_at_1(x):
    # todo 3 c)
    return


def build_F(t,N):
    # todo 3 d)
    return


def FEM_theta(N,M,theta):
    # todo 3 e)
    return


#### error analysis ####
nb_samples = 5
N = # fill in this line for f)-g)
M = # fill in this line for f)-g)
theta= # fill in this line for f)-g)


#### Do not change any code below! ####
l2error = np.zeros(nb_samples) 
k =  1 / M

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