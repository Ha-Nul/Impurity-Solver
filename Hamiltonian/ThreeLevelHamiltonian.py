import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cmath
import random

np.set_printoptions(threshold=784,linewidth=np.inf)

## prerequisites ##########################
def Standard_odd(r,z):

    A = [[0 for j in range(z)] for k in range(z)]

    for i in range(z):
        for j in range(z):
            try:
                if i==j:
                    A[i][j] = ((i+1)**2)
                if abs(i-j) == 1:
                    A[i][j] = -r/2
                else:
                    None
            except:
                None

    return np.array(A)

def Standard_odd_Eigenvec(r: float,z: int ,m: int):
    ''' Requires Standard_odd, m is mth eigenvector of H_0 Matrix, corresponds with mth eigenvalue'''

    A = Standard_odd(r,z)
    A_eigvec = np.linalg.eig(A)
    A_trans = np.transpose(A_eigvec[1])

    return A_trans[m]

def Standard_odd_Eigenval(r: float,z: int,m: int):
    ''' Requires Standard_odd, m it mth eigenvalue of H_0 Matrix, corresponds with mth eigenvector'''

    A = Standard_odd(r,z)
    A_eigval = np.linalg.eig(A)

    return A_eigval[0][m]

def Standard_even(r:float, z:int):
    '''Independent Function, Creates H_0 Hamiltonian in cosine basis
    r = Value of Gamma, z = Dimension of Matrix'''

    A = [[0 for j in range(z)] for k in range(z)]

    for i in range(z):
        for j in range(z):
            try:
                if i==j:
                    A[i][j] = (i)**2
                if abs(i-j) == 1:
                    A[i][j] = -r/2
                else:
                    None
            except:
                None

    np_A = np.array(A)
    np_A[0][1] = -r/(np.sqrt(2))
    np_A[1][0] = -r/(np.sqrt(2))

    return np_A

def Standard_even_Eigenvec(r: float,z: int ,m: int):
    ''' Requires Standard_even, m is mth eigenvector of H_0 Matrix, corresponds with mth eigenvalue'''

    A = Standard_even(r,z)
    A_eig = np.linalg.eig(A)
    A_trans = np.transpose(A_eig[1])

    return A_trans[m]

def Standard_even_Eigenval(r: float,z: int,m: int):
    ''' Requires Standard_even, m it mth eigenvalue of H_0 Matrix, corresponds with mth eigenvector'''

    A = Standard_even(r,z)
    A_eig = np.linalg.eig(A)

    return A_eig[0][m] 

def Elements(d,x,y):

    x_trans = np.transpose(x)

    arr = np.matmul(d,y)
    arr2 = np.matmul(x_trans,arr)

    return arr2

## Prerequisites #############################

## Main Hamiltonian ##########################

'''def Hamiltonian_Matrix(r: float,z: int,g: float,omega: float):
    Return Hamiltonian H in matrix form, requires prerequisites,
    r : Value of Gamma, z : dimension of Matrix, g : coupling strength, omega : frequency

    A = [[0 for j in range(z)] for k in range(z)]

    a_up = g*Elements(Standard_even(0,z),Standard_even_Eigenvec(r,z,0),Standard_even_Eigenvec(r,z,0))
    a_down = g*Elements(Standard_even(0,z),Standard_even_Eigenvec(r,z,1),Standard_even_Eigenvec(r,z,1))
    a_diagonal = g*Elements(Standard_even(0,z),Standard_even_Eigenvec(r,z,0),Standard_even_Eigenvec(r,z,1))


    for i in range(z):
        for j in range(z):
            try:

                if i==j and j%2 == 0:
                    A[i][j] = i*omega/2 + Standard_even_Eigenval(r,2,0)
                    A[i+1][j+1] = i*omega/2 + Standard_even_Eigenval(r,2,1)

                if (j-i) == 2 and j%2 == 0 or j%2 == 2: #파이썬은 행렬 index가 0부터 시작함.
                    A[i][j] = a_up*np.sqrt(j/2)
                    A[i+1][j+1] = a_down*np.sqrt(j/2)
                    A[i][j+1] = a_diagonal*np.sqrt(j/2)
                    A[i+1][j] = a_diagonal*np.sqrt(j/2)

                #if z%2 != 0 and i == z-2 and j == z-1 :
                    #A[i][j] = a_diagonal*np.sqrt(j/2)

                else:
                    None
            except:
                None

    np_A = np.array(A)
    np_B = np.array(A)
    sum_A = np.transpose(np_B)

    for i in range(z):
        for j in range(z):
            if i == j:
                sum_A[i][j] = 0

    test_A = sum_A + np_A


    return test_A
'''

def Hamiltonian_Matrix(r: float,z: int,g: float,omega: float):
    '''Return Hamiltonian H in matrix form, requires prerequisites,
    r : Value of Gamma, z : dimension of Matrix, g : coupling strength, omega : frequency '''

    #Matrix Elements
    eig_even_gs = Standard_even_Eigenval(r,2,0)
    eig_even_2n = Standard_even_Eigenval(r,2,1)
    eig_odd = Standard_odd_Eigenval(r,2,0)

    Diag_gs_to_1s = np.dot(Standard_odd_Eigenvec(r,z,0),Standard_even_Eigenvec(r,z,0))
    Diag_1s_to_2n = -np.dot(Standard_odd_Eigenvec(r,z,0),Standard_even_Eigenvec(r,z,1))

    #Matrix
    A = [[0 for i in range(z)] for j in range(z)]

    for i in range(z):
        for j in range(z):
            try:
                if i==j and i%3 == 0:
                   A[i][j+4] = (np.sqrt((j+4)//3))*g*Diag_gs_to_1s
                if i==j and i%3 == 1:
                    A[i][j+2] = (np.sqrt((j+2)//3))*g*Diag_gs_to_1s
                    A[i][j+4] = (np.sqrt((j+4)//3))*g*Diag_1s_to_2n
                if i==j and i%3 == 2:
                    A[i][j+2] = (np.sqrt((j+2)//3))*g*Diag_1s_to_2n
            except:
                None

    np_A = np.array(A)
    np_B = np.transpose(np_A.copy())
    '''
    for i in range(z):
        for j in range(z):
            try:
                if i==j and i%3==0:
                    A[i][j] = eig_even_gs+(j//3)
                elif i==j and i%3==1:
                    A[i][j] = eig_odd+(j//3)
                elif i==j and i%3==2:
                    A[i][j] = eig_even_2n+(j//3)
                else:
                    A[i][j] = 0
            except:
                None
    
    np_C = np.array(A)'''
    Array = np_A + np_B #+ np_C

    return Array

def Hamiltonian_Matrix_Eigenval(r,z,g,omega,i):
    A_eigval = np.linalg.eig(Hamiltonian_Matrix(r,z,g,omega))[0][i]
    return A_eigval

def Hamiltonian_Matrix_Eigenvec(r,z,g,omega,i):
    A = np.linalg.eig(Hamiltonian_Matrix(r,z,g,omega))[1].T
    A_eigvec = A[i]
    return A_eigvec


## Main Hamiltonian ##########################

## Pauil Matrix ##########################
'''
def Pauli(x):
    sigma = [[0 for i in range(2)] for j in range(2)]

    if x == 1:
        sigma[0][1] = 1
        sigma[1][0] = 1
        return np.array(sigma)
    if x == 2: 
        sigma[0][1] = 1j
        sigma[1][0] = -1j
        return np.array(sigma)
    if x == 3:
        sigma[0][0] = 1
        sigma[1][1] = 1
        return np.array(sigma)
    
def Pauli_tensorproduct(x,y):
    A = np.identity(y)
    Pauli_Tens = np.kron(A,Pauli(x))
    return Pauli_Tens
## Pauil Matrix ##########################

## Correlation function ##########################    
def Correlation_function(beta,r,tau,z,g):
    Correlation = []
    Z = []

    Inner_e_1 = sp.linalg.expm(-beta*Hamiltonian_Matrix_Eigenval(r,2*z,g,1))
    Inner_e_2 = sp.linalg.expm(tau*Hamiltonian_Matrix_Eigenval(r,2*z,g,1))
    Inner_e_3 = sp.linalg.expm(-tau*Hamiltonian_Matrix_Eigenval(r,2*z,g,1))

    matmul1 = np.matmul(Inner_e_3,Pauli_tensorproduct(1,z))
    matmul2 = np.matmul(Pauli_tensorproduct(1,z),matmul1)
    matmul3 = np.matmul(Inner_e_2,matmul2)
    matmul4 = np.matmul(Inner_e_1,matmul3)

    for i in range(2*z):
        Correlation.append(Elements(matmul4,Hamiltonian_Matrix_Eigenvec(r,2*z,g,1)[i],Hamiltonian_Matrix_Eigenvec(r,2*z,g,1)[i]))
        Z.append(sp.linalg.expm(-beta*Hamiltonian_Matrix_Eigenval(r,2*z,g,1)[i][i]))

    np_Cor = np.array(Correlation)
    np_Z = np.array(Z)
    
    sum_Cor = np.sum(np_Cor)
    sum_Z = np.sum(np_Z)

    return sum_Cor/sum_Z

def Spectral_Function(beta: float,r:float ,z: int,g: float,matsufreq: float,eta:float,omega: float):
    Spectral Density of Correlation_Function. r = Value of gamma, z = dimension/2 of Matrix,
    g = Coupling strength, matsufreq = Matsubara frequency of given Function, eta = Value for analytic continuation
    Zet = []
    for i in range(2):
        Zet.append(sp.linalg.expm(-beta*Standard_even(r,2))[i][i])
    Z_sum = np.sum(np.array(Zet))

    A = []
    
    for i in range(2*z):
        for j in range(2*z):
            expec = (Elements(Pauli_tensorproduct(1,z),Hamiltonian_Matrix_Eigenvec(r,2*z,g,omega,i),Hamiltonian_Matrix_Eigenvec(r,2*z,g,omega,j)))        
            conju = np.conjugate(expec)

            n = Hamiltonian_Matrix_Eigenval(r,2*z,g,omega,i)
            m = Hamiltonian_Matrix_Eigenval(r,2*z,g,omega,j)

            denom = (matsufreq + n - m)**2 + eta**2
            #sign of numerator e^E_m may can change 
            numer = np.exp(-beta*n)-np.exp(-beta*m)
            value = (conju*expec*numer*2*eta)/denom
            A.append(value)
            
    #np.imag() was not used.
    np_A = np.array(A)
    sum_A = np.sum(np_A)

    return sum_A/Z_sum
'''
## Lie group ####################################
def Lie_group(x):
    sigma = [[0 for i in range(3)] for j in range(3)]

    if x == 1:
        sigma[0][1] = 1
        sigma[1][0] = 1
        return np.array(sigma)
    if x == 2: 
        sigma[0][1] = 1j
        sigma[1][0] = -1j
        return np.array(sigma)
    if x == 3:
        sigma[0][0] = 1
        sigma[1][1] = 1
        return np.array(sigma)
    if x == 4:
        sigma[0][2] = 1
        sigma[2][0] = 1
        return np.array(sigma)
    if x == 5:
        sigma[0][2] = -1j
        sigma[2][0] = 1j
        return np.array(sigma)
    if x == 6:
        sigma[1][2] = 1
        sigma[2][1] = 1
        return np.array(sigma)
    if x == 7:
        sigma[1][2] = -1j
        sigma[2][1] = 1j
        return np.array(sigma)
    if x == 8:
        sigma[0][0] = 1/(3**0.5)
        sigma[2][2] = -2/(3**0.5)
        return np.array(sigma)

def Lie_tensorproduct(x,y):
    A = np.identity(y)
    Lie_Tens = np.kron(A,Lie_group(x))
    return Lie_Tens
## Lie group ####################################
## Spectral Function ############################
def Spectral_Function(beta: float,r:float ,z: int,g: float,matsufreq: float,eta:float,omega: float):
    '''Spectral Density of Correlation_Function. r = Value of gamma, z = dimension/2 of Matrix,
    g = Coupling strength, matsufreq = Matsubara frequency of given Function, eta = Value for analytic continuation'''
    
    # Tr Z
    eig_val = np.array([[Standard_even_Eigenval(r,3,0),0,0],
         [0,Standard_odd_Eigenval(r,3,0),0],
         [0,0,Standard_even_Eigenval(r,3,1)]])
    exp_val = sp.linalg.expm(-beta*eig_val)
    Z = np.sum(np.array(exp_val))

    # Spectral
    A = []
    for i in range(3*z):
        for j in range(3*z):
            expec_nm = Elements(Lie_tensorproduct(1,z),Hamiltonian_Matrix_Eigenvec(r,3*z,g,omega,i),Hamiltonian_Matrix_Eigenvec(r,3*z,g,omega,j))
            conju = np.conjugate(expec_nm)

            n = Hamiltonian_Matrix_Eigenval(r,3*z,g,omega,i)
            m = Hamiltonian_Matrix_Eigenval(r,3*z,g,omega,j)

            denom = (matsufreq + n - m)**2 + eta**2
            #sign of numerator e^E_m may can change 
            numer = np.exp(-beta*n)-np.exp(-beta*m)
            value = (conju*expec_nm*numer*2*eta)/denom

            #print(denom)

            A.append(value)
                    
    #np.imag() was not used.
    np_A = np.array(A)
    sum_A = np.sum(np_A)

    return sum_A/Z

## Spectral function ########################## 

## Chi_function ###############################

def Chi_sp(beta : float, r: float, omega: float, g: float, z: int , tau: float):
    '''Chi_function, r : value of gamma, z : dimension/3 of hilbert space, g : coupling strength, omega : frequency'''

    # Tr Z
    Exp_val = sp.linalg.expm(-beta*Hamiltonian_Matrix(r,3*z,g,omega))
    Z = np.trace(Exp_val)

    # Chi_calculation
    A = []
    for i in range(3*z):
        for j in range(3*z):
            E_N = Hamiltonian_Matrix_Eigenval(r,3*z,g,omega,i)
            E_M = Hamiltonian_Matrix_Eigenval(r,3*z,g,omega,j)
            Exp_val_chi = np.exp(-(beta-tau) * E_N - tau * E_M)

            Expec_NM = Elements(Lie_tensorproduct(1,z),Hamiltonian_Matrix_Eigenvec(r,3*z,g,omega,i),Hamiltonian_Matrix_Eigenvec(r,3*z,g,omega,j))

            A.append(Exp_val_chi * (Expec_NM**2))
    
    # Chi_total
    np_A = np.array(A)
    sum_A = np.sum(np_A)

    return sum_A/Z