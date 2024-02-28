import numpy as np
import matplotlib.pyplot as plt

# Time and step variables
T = 10       # Total time
dt = 0.001  # Time step size
N = int(T/dt) # total number of time steps

n = 100       # number of runs for the numerical solution

# Coefficients and Operators used in drift/diffusion terms
gamma = 0.01                     # coupling coefficient
A = np.array([[0,0], [1,0]])   # operator A = sigma minus 1 0, 0 -1
A_dagger = A.conj().T           # conj transpose of A
AAD = A + A_dagger              # A + A dagger

# Arbitrary Hamiltonian matrix
H = np.array([[0,1], [1,0]])   # H = sigma x
print("Hamiltonian matrix H is \n", H)

# Initial state psi
psi_0 = np.array([[0],[1]])     # psi is in the excited state
psi_0_dagger = psi_0.conj().T   # psi bra
rho_0 = psi_0 @ psi_0_dagger    # density matrix representing psi_0
print("Ψ(0) = \n", psi_0)
print("ρ(0) = \n", rho_0)

# Initialize time array
t_values = np.arange(0,T+dt,dt)

def EulerEvolution():

    # Initialize Y function array and Y(0) value
    psi_values = np.zeros((N+1, len(psi_0)), dtype=complex)
    psi_values[0] = psi_0.flatten()
    rho_values = np.zeros((N+1, 2, 2), dtype=complex)
    rho_values[0] = rho_0

    # Generate random increments for the Wiener process
    dW_values = np.sqrt(dt) * np.random.normal(0, 1, N+1)

    # Euler-Maruyama method to solve for future psi values
    # based on current psi state with drift and diffusion terms
    for i in range(N):

        # psi bra and ket calculation for drift/diffusion
        psi_ket = psi_values[i]
        psi_bra = psi_values[i].conj().T
        I = np.array([[1,0], [0,1]])

        # Calculate drift term (vector)

        schrodinger = ((-1j * H) @ psi_ket)

        term1 = (((psi_bra @ (AAD)) @ psi_ket) * A)

        term2 = (A_dagger @ A)

        term3 = (0.25 * ((((psi_bra @ (AAD)) @ psi_ket) **2) * I))

        drift = schrodinger + gamma/2 * ((term1 - term2 - term3) @ psi_ket)

        # Calculate diffusion term (vector)
        diffusion = np.sqrt(gamma) * ((A - 0.5 * ((psi_bra @ (AAD) @ psi_ket)) * I) @ psi_ket)

        new_psi = psi_values[i] + drift * dt + diffusion * dW_values[i]

        psi_values[i+1] = new_psi

        new_rho = new_psi.reshape(-1, 1) @ new_psi.reshape(-1, 1).conj().T

        rho_values[i+1] = new_rho


    return rho_values

# plotting each evolution and labeling each component

EvolutionList = {}

colorList = ["blue", "red"]

for i in range(0,n):

    EvolutionList[i] = EulerEvolution()
    currentEvolution = EvolutionList[i]

    if i==0:

        diagonal_elements = np.diagonal(currentEvolution.real, axis1=1, axis2=2) # diagonal elements (real) from each time step

        for j in range(currentEvolution.shape[1]): # plotting diagonal elements

            plt.plot(t_values, currentEvolution[:, j, j].real, label=f"ρ(t) {j}{j} ", color=colorList[j])
            #plt.plot(t_values, currentEvolution[:, 1 - j, j].real, label=f"ρ(t) {1 - j}{j}",color=colorList[j], linestyle='--')

        traces = np.zeros(N+1)

        for i in range(N+1):

            trace = np.trace(currentEvolution[i])
            traces[i] = trace

        plt.plot(t_values, traces, label="ρ(t) trace", color='black', linewidth=2)

    else:

        diagonal_elements = np.diagonal(currentEvolution.real, axis1=1,
                                        axis2=2)  # diagonal elements (real) from each time step

        for j in range(currentEvolution.shape[1]):  # plotting diagonal elements

            plt.plot(t_values, currentEvolution[:, j, j].real, label='_nolegend_', color=colorList[j])
            # plt.plot(t_values, currentEvolution[:, 1 - j, j].real, label=f"ρ(t) {1 - j}{j}",color=colorList[j], linestyle='--')

        traces = np.zeros(N + 1)

        for i in range(N + 1):
            trace = np.trace(currentEvolution[i])
            traces[i] = trace

        plt.plot(t_values, traces, label='_nolegend_', color='black', linewidth=2)

print(EvolutionList)

# plot labels and titles

plt.xlabel("Time")
plt.ylabel("ρ(t) Component")
plt.title("Stochastic Schrodinger Evolution of |1> by Euler Method (ρ(t) components)")
plt.legend()
plt.show()

# plotting the average for each component

AvgEvolution00 = np.zeros(N+1)
AvgEvolution11 = np.zeros(N+1)
AvgEvolutionTrace = np.zeros(N+1)
#AvgEvolution01 = np.zeros(N+1)
#AvgEvolution10 = np.zeros(N+1)

for i in range(N+1):
    sum00 = 0
    sum11 = 0
    sumTrace = 0
    #sum01 = 0
    #sum10 = 0
    for j in range(0,n):
        sum00 += EvolutionList[j][i][0][0]
        sum11 += EvolutionList[j][i][1][1]
        sumTrace += np.trace(EvolutionList[j][i])
        #sum01 += EvolutionList[j][i][1].imag
        #sum10 += np.sqrt(np.abs(EvolutionList[j][i][0])**2 + np.abs(EvolutionList[j][i][1])**2)
    AvgEvolution00[i] = sum00 / len(EvolutionList)
    AvgEvolution11[i] = sum11 / len(EvolutionList)
    AvgEvolutionTrace[i] = sumTrace / len(EvolutionList)
    #AvgEvolution01[i] = sum01 / len(EvolutionList)
    #AvgEvolution10[i] = sum10 / len(EvolutionList)


plt.plot(t_values, AvgEvolution00,label=f"Average ρ(t)00", color='green',linewidth=5)
plt.plot(t_values, AvgEvolution11,label=f"Average ρ(t)11 ",color='cyan',linewidth=5)
plt.plot(t_values, AvgEvolutionTrace,label="Average ρ(t) trace",color='gray',linewidth=5)
#plt.plot(t_values, AvgEvolution01,label=f"Average ρ(t)01",color='darkred',linewidth=5)
#plt.plot(t_values, AvgEvolution10,label=f"Average ρ(t)10",color='orange',linewidth=5)

# plot labels and titles

plt.xlabel("Time")
plt.ylabel("ρ(t) Component")
plt.title("Stochastic Schrodinger Evolution of |1> by Euler Method ρ(t)")
plt.legend()
plt.show()