import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin


# a) verify the solution
# u(x,t) = exp(-pi**2/4*t)*sin(pi/2*x)

# del_t u = exp(-pi**2/4*t) * (-pi**2/4) * sin(pi/2*x)

# del_x u = exp(-pi**2/4*t) * cos(pi/2*x) * pi/2

# del_xx u = del_x (exp(-pi**2/4*t) * cos(pi/2*x) * pi/2) = exp(-pi**2/4*t) * pi/2 * (-sin(pi/2*x)) * pi/2
# del_xx u = exp(-pi**2/4*t) * pi**2/4 * (-sin(pi/2*x))

# del_t u - del_xx u = exp(-pi**2/4*t) * (-pi**2/4) * sin(pi/2*x) - exp(-pi**2/4*t) * pi**2/4 * (-sin(pi/2*x))
# = - exp(-pi**2/4*t) * (pi**2/4) *sin(pi/2*x) + exp(-pi**2/4*t) * (pi**2/4) *sin(pi/2*x) = 0 => check

# u(x,0) = sin(pi/2*x) => check

# u(0,t) = exp(-pi**2/4*t)*sin(0) = 0 => check

# => all three equations of the PDE are satisfied by u

# Set how floating-point errors are handled.
np.seterr(all='raise')


def initial_value(x):
    return np.sin(np.pi/2 * x)


#### exact solution at t=1 ####
def exact_solution_at_1(x):
    return np.exp(-np.pi**2/4)*np.sin(np.pi/2*x)


#### numerical scheme ####
def eulerexplicit(N, M):
    G_mat = np.diag([-2]*N) + np.diag([1]*(N-1), k=1) + np.diag([1]*(N-1), k=-1)
    G_mat[N-1, N-2] = 2
    
    k = 1/M
    h = 1/N # N = 4 => h = 0.25
    nu = k/(h**2)

    C_mat = np.diag([1]*N) + nu*G_mat

    # N = 4 => x = [0, 0.25, 0.5, 0.75, 1] gives N = 4 intervalls
    x = np.linspace(0, 1, N+1)[1:]

    u = initial_value(x)

    for m in range(0,M):
        u = C_mat @ u # N = 4 => 4x4 @ 4
    return u


def eulerimplicit(N, M):
    G_mat = np.diag([-2]*N) + np.diag([1]*(N-1), k=1) + np.diag([1]*(N-1), k=-1)
    G_mat[N-1, N-2] = 2
    
    k = 1/M
    h = 1/N # N = 4 => h = 0.25
    nu = k/(h**2)

    C_mat = np.diag([1]*N) - nu*G_mat

    # N = 4 => x = [0, 0.25, 0.5, 0.75, 1] gives N = 4 intervalls
    x = np.linspace(0, 1, N+1)[1:]

    u = initial_value(x)

    for m in range(0,M):
        u = lin.solve(C_mat, u) # N = 4 => 4x4 @ 4
    return u


#### error analysis ####
nb_samples = 5
N = [2**l for l in range(2,2+nb_samples)]
M = [4**l for l in range(2,2+nb_samples)]
l2errorexplicit = np.zeros(nb_samples)  # error vector for explicit method
l2errorimplicit = np.zeros(nb_samples)  # error vector for implicit method
h2k = [1 / (N[i] ** 2) + 1 / M[i] for i in range(0, nb_samples)]

# comments: "overflow encountered in matmul" => approximation error unbounded. N = 4, M = 16 => h = 1/4, k = 1/16 => nu = k/h**2 = 1 but nu < 1/2 required for stability.


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