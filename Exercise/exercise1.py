import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin


# Set how floating-point errors are handled.
np.seterr(all='raise')


def initial_value(x):
    return np.sin(np.pi/2 * x)


#### exact solution at t=1 ####
def exact_solution_at_1(x):
    t=1
    return np.exp( -np.pi**2 /4 *t) * np.sin(np.pi/2 * x)

print(exact_solution_at_1(1))

#### numerical scheme ####
def eulerexplicit(N, M):
    k = 1/M
    h = 1/N
    nu = k/(h**2)

    G = (np.diag([-2]*N) + np.diag([1]*(N-1), k = 1) + np.diag([1]*(N-1), k = -1))
    G[N-1,N-2] = 2
    #print(G)
    
    u = np.zeros(shape=[N,M+1])
    u[:,0] = [initial_value(x*h) for x in range(1, N+1)]
    #print(u)

    for m in range(1,M+1):
        C = np.diag([1]*N) +nu*G
        
        #print(C)
        #print(u[:,(m-1)])
        
        u[:,m] = np.matmul( C, u[:,(m-1)])
        #print(u)

    return u[:,-1]

print(eulerexplicit(5,100))

def eulerimplicit(N, M):
    k = 1/M
    h = 1/N
    nu = k/(h**2)

    G = (np.diag([-2]*N) + np.diag([1]*(N-1), k = 1) + np.diag([1]*(N-1), k = -1))
    G[N-1,N-2] = 2
    #print(G)
    
    u = np.zeros(shape=[N,M+1])
    u[:,0] = [initial_value(x*h) for x in range(1, N+1)]
    #print(u)

    for m in range(1,M+1):
        C = np.diag([1]*N) - nu*G
        
        #print(C)
        #print(u[:,(m-1)])
        
        u[:,m] = np.linalg.solve(C, u[:,(m-1)])
        #print(u)

    return u[:,-1]

print(eulerimplicit(5,100))


#c and d

#Interpretation: we can see that for M_l = 2*4**(l+1) the plots for implicit and explicit method look almost idential. For M_l = 4**(l+1) the explicit method crashes the program.
# This is because there is the CFL-condition we covered in class: For k/h^2 <= 1/(2*(1-2*theta)). For the explicit method, theta = 0 so it simplifies to k/h^2 <= 1/2.
# When M_l = 2*4**(l+1), N_l = 2**(l+1) => k=1/(2*4**(l+1)), h = 1/(2**(l+1)) => k/(h^2) = (2**(l+1))**2 / (2*4**(l+1)) = 1/2 * (4**(l+1))/(4**(l+1)) = 1/2 => CLT criterion matched.
# Without the pre-factor of 2, this will not be the case => numerically unstable. In all other cases, by looking at the plots, you can see that error grows at most at the rate of O(h^2+k)

# #### error analysis ####
nb_samples = 5
N = [2**(l+1) for l in range(1,nb_samples+1)]
M = [2*4**(l+1) for l in range(1,nb_samples+1)] #M = [4**(l+1) for l in range(1,nb_samples+1)]
l2errorexplicit = np.zeros(nb_samples)  # error vector for explicit method
l2errorimplicit = np.zeros(nb_samples)  # error vector for implicit method
h2k = [1 / (N[j] ** 2) + 1 / M[j] for j in range(nb_samples)]
print(h2k)


#### Do not change any code below! ####
try:
    for i in range(nb_samples):
        l2errorexplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1)[1:]) - eulerexplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorexplicit), deg=1)
    if np.isnan(conv_rate[0]):
        raise Exception("Error unbounded for explicit method. Plots not shown.")
    print("Explicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: " + str(
        conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorexplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='$O(h^2+k)$')
    plt.title('$L^2$ convergence rate for explicit method', fontsize=13)
    plt.xlabel('$h^2+k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except Exception as e:
    print(f"Exception: {e}")

try:
    for i in range(nb_samples):
        l2errorimplicit[i] = (1 / N[i]) ** (1 / 2) * lin.norm(
            exact_solution_at_1(np.linspace(0, 1, N[i] + 1)[1:]) - eulerimplicit(N[i], M[i]), ord=2)
    conv_rate = np.polyfit(np.log(h2k), np.log(l2errorimplicit), deg=1)
    if np.isnan(conv_rate[0]):
        raise Exception("Error unbounded for implicit method. Plots not shown.")
    print("Implicit method converges: Convergence rate in discrete $L^2$ norm with respect to $h^2+k$: " + str(
        conv_rate[0]))
    plt.figure(figsize=[10, 6])
    plt.loglog(h2k, l2errorimplicit, '-x', label='error')
    plt.loglog(h2k, h2k, '--', label='$O(h^2+k)$')
    plt.title('$L^2$ convergence rate for implicit method', fontsize=13)
    plt.xlabel('$h^2+k$', fontsize=13)
    plt.ylabel('error', fontsize=13)
    plt.legend()
    plt.plot()
except Exception as e:
    print(f"Exception: {e}")

plt.show()