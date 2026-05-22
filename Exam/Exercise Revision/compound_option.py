import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
    

def solve_fem_two_stage(N, M, R, sigma, r, K, K1, T, T1, theta):
    

# Parameters
sigma = 
T = 
T1 = 
r = 
K = 
K1 = 
R = 
N = 
M = 

theta = 
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