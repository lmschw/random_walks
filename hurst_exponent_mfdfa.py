# Imports
from MFDFA import MFDFA
import numpy as np
import matplotlib.pyplot as plt

from random_walk_types.levy_walk import levy_walk
from random_walk_types.brownian_motion import brownian_motion_2d_without_sigma
from random_walk_types.correlated_random_walk import correlated_random_walk_2d

def compute_hurst_exponent(trajectory, lag, q, order):
    """ Compute the Hurst exponent using MFDFA.
        trajectory: 1D numpy array of the trajectory data"""
    lag, dfa = MFDFA(trajectory, lag = lag, q = q, order = order)

    plt.loglog(lag, dfa, 'o', label='fOU: MFDFA q=2')

    H_hat = np.polyfit(np.log(lag)[4:20],np.log(dfa[4:20]),1)[0]

    alpha = H_hat[0]

    if alpha >= 1:
        #print('Estimated H = '+'{:.3f}'.format(alpha-1))
        return alpha-1
    else:
        #print('Estimated H = '+'{:.3f}'.format(alpha))
        return alpha
    
def run_for_trajectory(trajectory):
    """ Run the Hurst exponent calculation for a given 2D trajectory. """
    lag = np.unique(np.logspace(0.5, 3, 100).astype(int))
    # Notice these must be ints, since these will segment
    # the data into chucks of lag size

    # q = 2 signifies that we want the classical Hurst exponent
    q = 2

    order = 1 # DFA

    H_x = compute_hurst_exponent(trajectory=trajectory[:,0], lag=lag, q=q, order=order)
    H_y = compute_hurst_exponent(trajectory=trajectory[:,1], lag=lag, q=q, order=order)

    print(f"H(x) = {H_x}, H(y) = {H_y}, H = {np.average([H_x, H_y])}")
    return H_x, H_y, np.average([H_x, H_y])

if __name__ == "__main__":
    """ Main function to run the Hurst exponent calculation on the test implementations of 
        different types of random walks. """
    t_final = 2000
    delta_t = 0.01

    time = np.arange(0, t_final, delta_t)

    # Distribution 
    DIST_TYPE = 3
    A = 2        
    B = 1
    steps = int(t_final / delta_t)
    print(f"STEPS: {steps}")
    #trajectory = levy_walk_simulation(steps, DIST_TYPE, A, B)
    trajectory_l = levy_walk(steps, A)
    trajectory_b = brownian_motion_2d_without_sigma(steps)
    trajectory_c = correlated_random_walk_2d(steps)

    print("Lévy walk")
    run_for_trajectory(trajectory_l)

    print("Brownian motion")
    run_for_trajectory(trajectory_b)

    print("CRW")
    run_for_trajectory(trajectory_c)