import math

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Time and step variables
T = 10        # Total time
dt = 0.001    # Time step size
N = int(T/dt) # total number of time steps
# Initialize time array
t_values = np.arange(0,T+dt,dt)

n = 1       # number of runs

# Coefficients and Operators
delta = 1
B = 0.5
gamma = 2

s_p = np.array([[0,1], [0,0]])
s_m = np.array([[0,0], [1,0]])
s_z = np.array([[1,0], [0,-1]])
I = np.array([[1,0],[0,1]])


# Hamiltonian matrix from Berislav Buˇca, Cameron Booker, and Dieter Jaksch
Hsum1 = np.kron(np.kron(s_p,s_m),I) + np.kron(np.kron(s_m,s_p),I) + delta * np.kron(np.kron(s_z,s_z),I) + B * np.kron(np.kron(I,s_z),I)
Hsum2 = np.kron(np.kron(s_m,I),s_p) + np.kron(np.kron(s_p,I),s_m) + delta * np.kron(np.kron(s_z,I),s_z) + B * np.kron(np.kron(s_z,I),I)
Hsum3 = np.kron(np.kron(I,s_p),s_m) + np.kron(np.kron(I,s_m),s_p) + delta * np.kron(np.kron(I,s_z),s_z) + B * np.kron(np.kron(I,I),s_z)

H = Hsum1 + Hsum2 + Hsum3

#H = np.array([[0,1,0,0], [1,0,0,0],[0,0,0,-1],[0,0,-1,0]])   # Hamiltonian = I x I
#H = np.array([[1,0], [0,-1]])   # Hamiltonian = sigma z
#H = np.kron(np.kron(H,I),I)

print("Hamiltonian matrix H is \n", H)

# V operator from Berislav Buˇca, Cameron Booker, and Dieter Jaksch
V = np.array([[0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]])   # from paper
#V = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
#V = np.array([[0,0], [1,0]])   # jump operator V = sigma minus
#V = np.kron(np.kron(V,I),I)


V_dag = V.conj().T # V dagger for Lindblad equation

# Initial state psi
rho_0 = np.array([[0.125,0,0,0,0,0,0,0], [0,0.125,0,0,0,0,0,0], [0,0,0.125,0,0,0,0,0], [0,0,0,0.125,0,0,0,0], [0,0,0,0,0.125,0,0,0], [0,0,0,0,0,0.125,0,0], [0,0,0,0,0,0,0.125,0], [0,0,0,0,0,0,0,0.125]]) # density matrix for ground state |111>
#rho_0 = np.array([[1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]) # density matrix for excited state |00>
#rho_0 = np.array([[1,0], [0,0]]) # density matrix for excited state |0>
print("ρ(0) = \n", rho_0)

rho_0 = rho_0.reshape(len(H)*len(H), 1)

rho_0 = sp.Matrix(rho_0)

def Lindblad_Eq():

    # Solved Lindblad equation based on jump operators

    # dimensions of jump operator V
    rows, cols = V.shape

    # Create a square matrix rho with symbols
    rho = sp.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            rho[i, j] = sp.Symbol(f'rho{i}{j}')

    print(rho)

    A = -1j * (H @ rho - rho @ H) + gamma * (V @ rho @ V_dag - 0.5*(V_dag @ V @ rho) - 0.5*(rho @ V_dag @ V))

    print("A: ", A)

    ## Convert A to a 4x4 with rho as a 4x1 column vector

    variables = [sp.Symbol(f'rho{i}{j}') for i in range(rows) for j in range(cols)]

    # initialize M (then append each row)

    M = np.empty((0, len(A)))

    for j in range(len(H)):

        for k in range(len(H)):

            # step 1: create a polynomial eq obj for each elem of A

            poly_expr = sp.Poly(A[j, k], *variables)

            # step 2: Get monoms() and coeffs() for the poly

            vars_present = poly_expr.monoms()

            coeffs = poly_expr.coeffs()

            # initialize the new row for m

            m_row = [0 for _ in range(len(A))]

            # for each coeff, put it in the row based on the vars location

            for i in range(len(coeffs)):
                var_index = np.asarray(vars_present[i])
                each_coeff = np.asarray(coeffs[i])

                m_row += each_coeff * var_index

                print("one coeff * var index: ", each_coeff * var_index)

            print("m_row: ", m_row)

            # append m_row to M matrix

            M = np.vstack((M, m_row))

    M = sp.Matrix(M)

    print("M: ", M)

    P, D = M.diagonalize()

    D_numpy = np.array(D)

    print("Diagonalized Matrix of M: ", D)

    for i in range(len(D_numpy)):

        if (sp.Abs(sp.re(D_numpy[i][i])) <= 1*10**(-10)):
            D_numpy[i][i] = 0 + sp.im(D_numpy[i][i])*sp.I

        if (sp.Abs(sp.im(D_numpy[i][i])) <= 1*10**(-10)):
            D_numpy[i][i] = sp.re(D_numpy[i][i]) + 0*sp.I

        print("EValue: ", D_numpy[i][i])

    P_np = np.array(P)

    for i in range(len(P_np)):

        for j in range(len(P_np)):

            if (sp.Abs(sp.re(P_np[i][j])) <= 1*10**(-10)):
                P_np[i][j] = 0 + sp.im(P_np[i][j])*sp.I

            if (sp.Abs(sp.im(P_np[i][j])) <= 1*10**(-10)):
                P_np[i][j] = sp.re(P_np[i][j]) + 0*sp.I


    P = sp.Matrix(P_np)
    D = sp.Matrix(D_numpy)

    print(P)

    P_inv = P.inv()

    t = sp.Symbol('t')

    DExp = sp.diag(*[sp.exp(D[i, i]*t) for i in range(D.shape[0])]) #.exp(D)

    print("DExp: ", DExp)

    rho_t = P @ DExp @ P_inv

    # function to convert rho_t to an sp.matrix with sp.functions inside
    def convert_elements_to_functions(matrix, variable):
        def to_function(element):
            # Use element's string representation to create a new sympy Function
            return sp.Function(str(element))(variable)

        return matrix.applyfunc(to_function)

    print("rho_t: ", rho_t)

    constants = sp.Matrix(sp.symbols('c1:{}'.format(len(H)*len(H)+1)))

    equation1 = sp.Eq(rho_0, rho_t.evalf(subs={t: 0}) @ constants)

    print("Equation1: ", equation1)

    solution = sp.solve(equation1, constants)

    print("Solution: ", solution)

    evoFunction = (rho_t @ constants).subs(solution).reshape(len(H), len(H))

    rho0 = sp.Matrix(len(H), len(H), sp.symbols(f'p1:{len(H)*len(H) + 1}'))

    equation2 = sp.Eq(rho0,evoFunction)

    print("Equation 2: ", equation2)

    # V = sigma minus, np.array([[0,0], [1,0]])
    # and hamiltonian H = sigma z
    # dp/dt = M --> P(t) = ADA^-1 (constants)
    # Solved equation: P(t) = ADA^-1 (constants)
    # where A = eigenvectors of M
    # and D = e^(lambda*t)... eigenvalues of M

    # Create lambdified functions for each element of the matrix
    rho_functions = [sp.lambdify(t, elem) for elem in evoFunction]

    # Evaluate the lambdified functions for each time value
    rho_values = np.array([[func(t_val) for func in rho_functions] for t_val in t_values])

    # Plot the values of each element of rho against time
    plt.figure(figsize=(10, 6))

    j = 26

    for i in range(rho_values.shape[1]):
        if i % ((len(H) + 1)) == 0:  # Check if the index is a diagonal element
            plt.plot(t_values, rho_values[:, i], label=f"Diagonal Element {i + 1}", linewidth=j)
            j -= 3
        #plt.plot(t_values, rho_values[:, i], label=f"Element {i + 1}")

    plt.xlabel('Time')
    plt.ylabel('ρ element magnitude')
    plt.title('3Q Lindblad Evolution of |111> ρ ')
    plt.legend()
    plt.grid(True)
    plt.show()

    ## Plot 2

    # Plot the values of each element of rho against time
    plt.figure(figsize=(10, 6))

    for i in range(rho_values.shape[1]):
        if i % ((len(H) + 1)) == 0:  # Check if the index is a diagonal element
            plt.plot(t_values, rho_values[:, i], label=f"Diagonal Element {i + 1}")
        # plt.plot(t_values, rho_values[:, i], label=f"Element {i + 1}")

    plt.xlabel('Time')
    plt.ylabel('ρ element magnitude')
    plt.title('3Q Lindblad Evolution ρ ')
    plt.legend()
    plt.grid(True)
    plt.show()



Lindblad_Eq = Lindblad_Eq()