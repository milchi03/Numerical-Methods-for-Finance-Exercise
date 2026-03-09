import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


#K(x,y) = (1/2 * x**2 + x |x = x,y) = 1/2 * y**2 + y - 1/2 * x**2 - x
def kappa_integral(x,y):
    return 1/2 * y**2 + y - 1/2 * x**2 - x


def build_massMatrix(N):
    return  np.diag([1]*N) #bc integral (dirac * dirac = 1 if i=j, 0 else)

print(build_massMatrix(3))

def build_rigidityMatrix(N):
    # todo 3 b)
    # Be careful with the indices!
    # kappa_integral could be helpful here
    # phi'_i(x) = 1/h on (i-1,i) and -1/h on (i,i+1)
    h = 1/N  
    A = np.zeros((N,N))
    
    x = np.linspace(0, 1, N+2)
    print(x)

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

print(build_rigidityMatrix(3))

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
N = 1# fill in this line for f)-g)
M = 1# fill in this line for f)-g)
theta= 1# fill in this line for f)-g)


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