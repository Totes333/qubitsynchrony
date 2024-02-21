import numpy as np
import matplotlib.pyplot as plt



# Time and step variables
T = 10       # Total time
dt = 0.01  # Time step size
N = int(T/dt) # total number of time steps

n = 20       # number of runs for the numerical solution

# Coefficients and Operators used in drift/diffusion terms
gamma = 0.01                     # coupling coefficient
A = np.array([[1,0], [0,-1j]])   # operator A = sigma minus 1 0, 0 -1
A_dagger = A.conj().T           # conj transpose of A
AAD = A + A_dagger              # A + A dagger

# Arbitrary Hamiltonian matrix
H = np.array([[0,1], [1,0]])   # H = sigma x
print("Hamiltonian matrix H is \n", H)

# Initial state psi
psi_0 = np.array([[0],[1]])     # psi is in the excited state
print("Ψ(0) = \n", psi_0)

# Initialize time array
t_values = np.arange(0,T+dt,dt)

def EulerEvolution():

    # Initialize Y function array and Y(0) value
    psi_values = np.zeros((N+1, len(psi_0)), dtype=complex)
    psi_values[0] = psi_0.flatten()

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

        #drift = schrodinger + gamma/2 * (((((psi_bra @ (AAD)) @ psi_ket) * A) - (A_dagger @ A) - (0.25 * ((((psi_bra @ (AAD)) @ psi_ket) **2) * I))) @ psi_ket)

        drift = schrodinger + gamma/2 * ((term1 - term2 - term3) @ psi_ket)

        # Calculate diffusion term (vector)
        diffusion = np.sqrt(gamma) * ((A - 0.5 * (psi_bra @ (AAD) @ psi_ket)) @ psi_ket)

        psi_values[i+1] = psi_values[i] + drift * dt + diffusion * dW_values[i]

    return psi_values

# plotting each evolution and labeling each component

EvolutionList = {}
colorList = ["blue", "red"]

for i in range(0,n):

    EvolutionList[i] = EulerEvolution()
    currentEvolution = EvolutionList[i]

    if i==0:
        for i in range(currentEvolution.shape[1]):
            plt.plot(t_values, currentEvolution[:, i].real, label=f"Ψ(t) component {i + 1} real", color=colorList[i])
            plt.plot(t_values, currentEvolution[:, i].imag, label=f"Ψ(t) component {i + 1} imaginary",
                     color=colorList[i], linestyle='--')

        magnitude = np.sqrt(np.sum(np.abs(currentEvolution) ** 2, axis=1))
        plt.plot(t_values, magnitude, label="|Ψ(t)|", color='black', linewidth=2)

    else:

        for i in range(currentEvolution.shape[1]):
            plt.plot(t_values, currentEvolution[:, i].real, label='_nolegend_', color=colorList[i])
            plt.plot(t_values, currentEvolution[:, i].imag, label='_nolegend_',
                     color=colorList[i], linestyle='--')

        magnitude = np.sqrt(np.sum(np.abs(currentEvolution) ** 2, axis=1))
        plt.plot(t_values, magnitude, label='_nolegend_', color='black', linewidth=2)

# plot labels and titles

"""plt.xlabel("Time")
plt.ylabel("Wavefunction Component Magnitude")
plt.title("Stochastic Schrodinger Evolution of |1> by Euler Method")
plt.legend()
plt.show()
"""

# plotting the average for each component

AvgEvolution1Real = np.zeros(N+1)
AvgEvolution1Imag = np.zeros(N+1)
AvgEvolution2Real = np.zeros(N+1)
AvgEvolution2Imag = np.zeros(N+1)
AvgEvolutionMag = np.zeros(N+1)

for i in range(N+1):
    sum1Real = 0
    sum1Imag = 0
    sum2Real = 0
    sum2Imag = 0
    sumMag = 0
    for j in range(0,n):
        sum1Real += EvolutionList[j][i][0].real
        sum1Imag += EvolutionList[j][i][0].imag
        sum2Real += EvolutionList[j][i][1].real
        sum2Imag += EvolutionList[j][i][1].imag
        sumMag += np.sqrt(np.abs(EvolutionList[j][i][0])**2 + np.abs(EvolutionList[j][i][1])**2)
    AvgEvolution1Real[i] = sum1Real / len(EvolutionList)
    AvgEvolution1Imag[i] = sum1Imag / len(EvolutionList)
    AvgEvolution2Real[i] = sum2Real / len(EvolutionList)
    AvgEvolution2Imag[i] = sum2Imag / len(EvolutionList)
    AvgEvolutionMag[i] = sumMag / len(EvolutionList)


plt.plot(t_values, AvgEvolution1Real,label=f"Average Ψ(t) component 1 Real", color='green',linewidth=5)
plt.plot(t_values, AvgEvolution1Imag,label=f"Average Ψ(t) component 1 Imaginary",color='cyan',linewidth=5)
plt.plot(t_values, AvgEvolution2Real,label=f"Average Ψ(t) component 2 Real",color='orange',linewidth=5)
plt.plot(t_values, AvgEvolution2Imag,label=f"Average Ψ(t) component 2 Imaginary",color='darkred',linewidth=5)
plt.plot(t_values, AvgEvolutionMag,label="Average |Ψ(t)|",color='gray',linewidth=5)

# plot labels and titles

plt.xlabel("Time")
plt.ylabel("Wavefunction Component Magnitude")
plt.title("Stochastic Schrodinger Evolution of |1> by Euler Method")
plt.legend()
plt.show()