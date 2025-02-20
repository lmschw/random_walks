# Imports
from MFDFA import MFDFA
from MFDFA import fgn
import numpy as np
import matplotlib.pyplot as plt

from levy_walk import levy_walk_simulation
#from levy_walk_2 import levy_walk
from levy_walk_3 import levy_walk_3
from brownian_motion import brownian_motion_2d_without_sigma

def compute_hurst_exponent(trajectory, lag, q, order):
    lag, dfa = MFDFA(trajectory, lag = lag, q = q, order = order)

    plt.loglog(lag, dfa, 'o', label='fOU: MFDFA q=2')

    H_hat = np.polyfit(np.log(lag)[4:20],np.log(dfa[4:20]),1)[0]

    alpha = H_hat[0]

    if alpha >= 1:
        print('Estimated H = '+'{:.3f}'.format(alpha-1))
        return alpha-1
    else:
        print('Estimated H = '+'{:.3f}'.format(alpha))
        return alpha

t_final = 2000
delta_t = 0.001

time = np.arange(0, t_final, delta_t)

# Distribution 
DIST_TYPE = 3
A = 2        
B = 1
steps = int(t_final / delta_t)
print(f"STEPS: {steps}")
#trajectory = levy_walk_simulation(steps, DIST_TYPE, A, B)
trajectory = levy_walk_3(steps, A)
#trajectory = brownian_motion_2d_without_sigma(steps)

lag = np.unique(np.logspace(0.5, 3, 100).astype(int))
# Notice these must be ints, since these will segment
# the data into chucks of lag size

# q = 2 signifies that we want the classical Hurst exponent
q = 2

order = 1 # DFA

H_x = compute_hurst_exponent(trajectory=trajectory[:,0], lag=lag, q=q, order=order)
H_y = compute_hurst_exponent(trajectory=trajectory[:,1], lag=lag, q=q, order=order)

print(f"H(x) = {H_x}, H(y) = {H_y}, H = {np.average([H_x, H_y])}")