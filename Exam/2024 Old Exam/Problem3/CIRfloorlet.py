from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from assembleMatrix import assembleMatrix


def buildMassCIR(N, R, mu):
    return


def buildACIR(N, R, alpha, beta, sigma, mu):
    return


def g(s, T, T1, K):
    return


def FEM_theta(N, k, R, t, alpha, beta, sigma, mu, u_init, theta):
    return


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
    alpha = 0.05 * 0.2
    beta = 0.2
    sigma = 0.1

    N = 2**9 - 1
    k = T1 / (N + 1)
    theta = 0.5
    mu = -0.1

    # Compute zero coupon bond price
    # Initial condition
    u_init = 
    # zero coupon bond price at time T
    u_0 = 

    # Compute floorlet price
    u_1 = 

    ##################### You do not need to modify any code below this line #####################

    # Plot the zero coupon bond price
    fig_1, ax_1 = plt.subplots()
    plot_FEM(fig_1, ax_1, 4, u_0, N, R, r"$V_0(r, T)$", 0)
    ax_1.set_title(r"Zero coupon bond price at $T=2$")
    ax_1.set_xlabel(r"$r$")
    ax_1.set_ylabel("Price")
    ax_1.legend()
    plt.savefig(Path.home() / "questions" / "Problem3" / "plot_bond.pdf", format="pdf")
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
    plt.savefig(Path.home() / "questions" / "Problem3" / "plot_floorlet.pdf", format="pdf")
    plt.show()
