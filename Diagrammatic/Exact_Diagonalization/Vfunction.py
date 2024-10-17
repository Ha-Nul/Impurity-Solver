import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cmath
import random

np.set_printoptions(threshold=784,linewidth=np.inf)

## Full source code available : 2023 09 06.ipynb

## K sum prerequisites ####################

k = np.full(10,1)
tau = np.linspace(0,0.5,10)

## k linspace and tau grid requried #################

b = 2

#unit = 1e-21 

def bose_dist(x):

    T = 273
    #boltz = ct.k*Ts

    return 1/(np.exp(x*b)-1)

def green(tau,k):

    #for i in range(len(tau)):
        #if tau[i] > 0:
            #return (bose_dist(k)+1)*np.exp(-k*tau)
        #if tau[i] < 0:
            #return (bose_dist(k))*np.exp(-k*tau)
        return ((bose_dist(k)+1)*np.exp(-k*tau)) + (bose_dist(k))*np.exp(k*tau)

def omega(v):
    return v*np.abs(k)

def coupling(v,g,W):
    w = omega(v)
    cut_off = W
    return g*np.sqrt(w/(1+(w/cut_off)**2))

def interact(tau,v,g,W):

    g_k = np.abs(coupling(v,g,W))**2

    n = len(k)

    k_sum = np.zeros(n)
    t_array = np.zeros(n)

    for j in range(n):
        t = tau[j]
        for i in range(n):
            k_sum[i] = g_k[i] * green(t,omega(v))[i]
        t_array[j] = np.sum(k_sum)
        k_sum = np.zeros(len(k))
    
    #print(t_array)
    return t_array

def output(x):
    """numpy를 이용해 데이터를 dat 형식으로 출력하며, x에는 주어지는 tau값만 넣어서 사용하는 함수입니다."""
    import numpy as np

    a = x
    b = interact(x)

    df = np.column_stack((a,b))

    np.savetxt('Vdat.dat',df)
    
    return None