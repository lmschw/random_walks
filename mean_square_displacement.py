import numpy as np
from scipy.optimize import curve_fit

from random_walk_types.levy_walk import levy_walk
from random_walk_types.brownian_motion import brownian_motion_2d_without_sigma
from random_walk_types.correlated_random_walk import correlated_random_walk_2d

def msd(trajectory, DT=0.05):
    """ Calculate the Mean Squared Displacement (MSD) of a 2D trajectory. """
    sampling_interval = int(1/DT) 
    sampled_trajectory = trajectory[::sampling_interval]  # Sample the trajectory every simulated second
    # print(sampled_trajectory)
    
    msd = np.zeros(len(sampled_trajectory)-1)
    for i in range(1, len(sampled_trajectory)):
        differences = sampled_trajectory[i:] - sampled_trajectory[:-i]
        squared_displacements = np.square(differences).sum(axis=1)
        msd[i-1] = np.mean(squared_displacements)

    return msd

# Fit a power-law function to the MSD data to extract the scaling exponent (α)
def power_law(x, a):
    return x**a/2

def run_for_trajectory(trajectory, name):
    """ Run the MSD calculation for a given 2D trajectory. """
    msds = msd(trajectory, 1)

    print(f"{name}: {msds[-1]}")

    popt, pcov = curve_fit(power_law, range(1, len(msds) + 1), msds)
    alpha = popt[0]
    print(f"Scaling exponent (α) {name}: {alpha}")
    print(f"MSD at last step {name}: {msds[-1]}")
    return msds[-1], alpha

if __name__ == "__main__":
    """ Main function to run the MSD calculation on the test implementations of
        different types of random walks. """
    STEPS = 10000

    DIST_TYPE = 3
    A = 1.5         
    B = 1
    print("Lévy")
    #trajectory_l = levy_walk_simulation(STEPS, DIST_TYPE, A, B)
    trajectory_l = levy_walk(STEPS, A)
    run_for_trajectory(trajectory=trajectory_l, name='Lévy walk')

    print("Brownian")
    trajectory_b = brownian_motion_2d_without_sigma(STEPS)
    run_for_trajectory(trajectory=trajectory_b, name='Brownian motion')

    print("CRW")
    trajectory_c = correlated_random_walk_2d(STEPS)
    run_for_trajectory(trajectory=trajectory_c, name='CRW')