from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from assembleMatrix import assembleMatrix


def buildMassCIR(N, R, mu):
    def a(r):
        return 0 * r #wth?
    def b(r):
        return 0 * r #wth?
    def c(r):
        return r**(2*mu)
    
    M_mat = assembleMatrix(N, R, a, b, c)
    return M_mat


def buildACIR(N, R, alpha, beta, sigma, mu):
    def a(r):
        return 1/2 * sigma**2 * r**(2*mu+1)
    def b(r):
        return sigma**2 * (1/2 + mu) * r**(2*mu) + (beta*r - alpha) * r**(2*mu)
    def c(r):
        return r**(2*mu+1)
    
    A_mat = assembleMatrix(N, R, a, b, c)
    return A_mat


def g(s, T, T1, K):
    return


def FEM_theta(N, k, R, t, alpha, beta, sigma, mu, u_init, theta):
    #h = R/(N+1)
    M = int(np.floor(t/k)) #k??
    #k = 1/M #k??

    #x = np.array([i*h for i in range(1, N+1)])

    M_mat = buildMassCIR(N, R, mu)
    A_mat = buildACIR(N, R, alpha, beta, sigma, mu)

    B_mat = M_mat + k*theta*A_mat
    C_mat = M_mat - k*(1-theta)*A_mat

    u_sol = u_init

    for _ in range(M):
        RHS = C_mat @ u_sol
        u_sol = spsolve(B_mat, RHS)

    return u_sol


def plot_FEM(fig, ax, rightlim, u_sol, N, R, label, markersize=2):
    grid = np.linspace(0, R, N + 2)[:-1]
    I = grid <= rightlim
    ax.plot(grid[I], u_sol[I], "rx-", label=label, markersize=markersize, linewidth=0.5)


if __name__ == "__main__":
    # Parameters
    T = 2
    T1 = 4
    K = 0.05
    R = 4
    alpha = 0.01
    beta = 0.2
    sigma = 0.1

    N = 2**9 - 1
    k = T1 / (N + 1)
    theta = 0.5
    mu = -0.1

    # Compute zero coupon bond price
    # Initial condition
    u_init = np.array([1. for _ in range(N+1)])

    # zero coupon bond price at time T
    u_0 = FEM_theta(N, k, R, T1-T, alpha, beta, sigma, mu, u_init, theta)

    # Compute floorlet price
    def g1(V_0, T, T1):
        g1_tilde = (T1 - T) * np.maximum(0, K - (1-V_0) / ((T1-T)*V_0))
        return V_0 * g1_tilde

    call_payoff = g1(u_0, T, T1)
    plt.plot(u_0, call_payoff) #type: ignore
    plt.show()
    u_1 = FEM_theta(N, k, R, T, alpha, beta, sigma, mu, call_payoff, theta)

    ##################### You do not need to modify any code below this line #####################

    # Plot the zero coupon bond price
    fig_1, ax_1 = plt.subplots()
    plot_FEM(fig_1, ax_1, 4, u_0, N, R, r"$V_0(r, T)$", 0)
    ax_1.set_title(r"Zero coupon bond price at $T=2$")
    ax_1.set_xlabel(r"$r$")
    ax_1.set_ylabel("Price")
    ax_1.legend()
    #plt.savefig(Path.home() / "questions" / "Problem3" / "plot_bond.pdf", format="pdf")
    plt.show(block=False)

    from exact import circpl_CIR

    grid = np.linspace(0, R, 1000)[:-1]
    I = grid <= 0.1
    exact = circpl_CIR(grid[I], alpha, beta, sigma, T, T1, K)

    # Plot the floorlet price
    fig_2, ax_2 = plt.subplots()
    ax_2.plot(
        grid[I], exact, label=r"Exact", linewidth=1.5, linestyle="dashed", color="black"
    )
    plot_FEM(fig_2, ax_2, 0.1, u_1, N, R, r"$V_1(r, t)$")
    ax_2.set_title(r"Floorlet price at $T=2$")
    ax_2.set_xlabel(r"$r$")
    ax_2.set_ylabel("Price")
    ax_2.legend()
    #plt.savefig(Path.home() / "questions" / "Problem3" / "plot_floorlet.pdf", format="pdf")
    plt.show()
