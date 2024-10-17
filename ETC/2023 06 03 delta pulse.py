import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cn

def Function(t):
    n = 10
    return (1/(+0.01*t))*np.sin(n*t)

jmax = np.linspace(0,10,1000)

for i in range(1000):
    
    if i%10 == 0:
        plt.plot(jmax,Function(jmax-jmax[i+1]))
        plt.ylim(-1000,1000)
        plt.xlim(0,10)
        plt.show()
    