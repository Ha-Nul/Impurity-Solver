import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as ct

eps0 = ct.epsilon_0
mu0 = ct.mu_0
c0 = ct.speed_of_light
imp0 = np.sqrt(mu0/eps0)

jmax = 500
jsource = 10
nmax = 2000

Ex = np.zeros(jmax)
Hz = np.zeros(jmax)
Ex_prev = np.zeros(jmax)
Hz_prev = np.zeros(jmax)

lambda_min = 350e-9
dx = lambda_min/20
dt = dx / c0

eps = np.ones(jmax) * eps0
eps[250:300] = 10*eps0

material_prof = eps > eps0

def Source_Function(t):
    lambda_0 = 550e-9
    w0 = 2*np.pi*c0/lambda_0
    tau = 30
    t0 = tau*3
    
    return np.exp(-(t-t0)**2/tau**2)*np.sin(w0*t*dt)


for n in range(nmax):
    #update magnetic field boundaries
    Hz[jmax-1] = Hz_prev[jmax-2]
    
    for j in range(jmax-1):
        Hz[j] = Hz_prev[j] + dt/(dx*mu0) * (Ex[j+1] - Ex[j])
        Hz_prev[j] = Hz[j]
    #Magenetic field source
    Hz[jsource-1] -= Source_Function(n)/imp0
    Hz_prev[jsource - 1] = Hz[jsource -1]

    #update magnetic field boundaries
    Ex[0] = Ex_prev[1] 
    #Update electric field source
    for j in range(1,jmax):
        Ex[j] = Ex_prev[j] + dt/(dx*eps[j]) * (Hz[j] - Hz[j-1])
        Ex_prev[j] = Ex[j]
    #Electric field source
    Ex[jsource] += Source_Function(n+1)
    Ex_prev[jsource] = Ex[jsource]

    if n%10 == 0:
        plt.plot(Ex)
        plt.plot(material_prof)
        plt.ylim([-1,1])
        plt.show()
        plt.close()
        