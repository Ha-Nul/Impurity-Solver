import numpy as np
import scipy as sp

gamma = 1
alpha = 0.5
cutoff = 20
omega = 1

mode_grid = np.linspace(1,100,100)
tau_grid = np.linspace(0,1,200)

beta = tau_grid[len(tau_grid)-1]
M = mode_grid[len(mode_grid)-1]

matsu_freq_grid = np.linspace(0,1,200)