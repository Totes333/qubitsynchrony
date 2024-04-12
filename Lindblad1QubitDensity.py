import math

import numpy as np
import matplotlib.pyplot as plt

# Time and step variables
T = 10        # Total time
dt = 0.001    # Time step size
N = int(T/dt) # total number of time steps


n = 1       # number of runs

# Arbitrary Hamiltonian matrix
#H = np.array([[0,1], [1,0]])   # Hamiltonian = sigma x
H = np.array([[1,0], [0,-1]])   # Hamiltonian = sigma z
print("Hamiltonian matrix H is \n", H)

# Initial state psi
rho_0 = np.array([[1,0], [0,0]]) # density matrix for ground state |0>
print("ρ(0) = \n", rho_0)

# Initialize time array
t_values = np.linspace(0,T+dt,N+1)

def Lindblad_Evolve():

    # Solved Lindblad equation based on jump operators
    # V = sigma minus, np.array([[0,0], [1,0]])
    # and hamiltonian H = sigma z
    # dp/dt = M --> P(t) = ADA^-1 (constants)
    # Solved equation: P(t) = ADA^-1 (constants)
    # where A = eigenvectors of M
    # and D = e^(lambda*t)... eigenvalues of M

    rho_t = np.empty([N+1,2,2])

    for i in range(N+1):

        rho_t[i] = [[np.e**(-t_values[i]), 0],[0,1 - np.e**(-t_values[i])]]

    return rho_t



Lindblad_Evolution = Lindblad_Evolve()
print(Lindblad_Evolution)

# plotting Lindblad Evolution and labeling each component

colorList = ["blue","red", "green", "orange"]

for i in range(2):

    plt.plot(t_values, Lindblad_Evolution[:, i, i].real, label=f"ρ(t) {i}{i} ", color=colorList[i]) # diagonal elements (checked that they were real)
    plt.plot(t_values, Lindblad_Evolution[:, i, 1 - i].real, label=f"ρ(t) {1 - i}{i} real",color=colorList[i+2]) # off-diagonal (checked that they were real)


traces = np.zeros(N+1)

for i in range(N+1):

    trace = np.trace(Lindblad_Evolution[i])
    traces[i] = trace

plt.plot(t_values, traces, label="ρ(t) trace", color='black', linewidth=2)

# plot labels and titles

plt.xlabel("Time")
plt.ylabel("ρ(t) Element Magnitude")
plt.title("Lindblad Evolution of |0> ρ(t)")
plt.legend()
plt.show()

