import numpy as np
import matplotlib.pyplot as plt

# Parameters
drift = 0.2
diffusion = 0.1

Y_0 = 1  # Initial value of function Y
T = 1   # Total time
N = 1000  # Number of time steps
dt = T / N  # Time step size
n = 20 # number of runs for the numerical solution

# Initialize arrays to store results
t_values = np.arange(0, T + dt, dt)

def EulerEvolution():

    # Initialize Y function array and Y(0) value
    Y_values = np.zeros(N+1)
    Y_values[0] = Y_0


    # Generate random increments for the Wiener process
    dW_values = np.sqrt(dt) * np.random.normal(0, 1, N)

    # Euler-Maruyama method to solve for future state
    # based on current state with drift and diffusion terms
    for i in range(N):
        Y_values[i+1] = Y_values[i] + drift * dt + diffusion * dW_values[i]

    return Y_values

# Plot each evolution

EvolutionList = {}
AvgEvolution = np.zeros(N+1)

for i in range(0,n):

    EvolutionList[i] = EulerEvolution()
    currentEvolution = EvolutionList[i]

    plt.plot(t_values, currentEvolution)

# Calculate and plot the average of n evolutions

for i in range(N+1):
    sum = 0
    for j in range(0,n):
        sum += EvolutionList[j][i]
    AvgEvolution[i] = sum/len(EvolutionList)

plt.plot(t_values, AvgEvolution,color='black',linewidth=3)

plt.xlabel('Time')
plt.ylabel('Function Y')
plt.title('Euler-Maruyama Numerical Solution of dY = drift*dt + diffusion*dW')
plt.show()
