import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from levy_walk import levy_walk_simulation
from levy_walk_2 import levy_walk
from levy_walk_3 import levy_walk_3
from brownian_motion import brownian_motion_2d, brownian_motion_2d_without_sigma

def compute_MSD(positions, num_steps):
   msds=[0]
   for i in range(1, num_steps):
       msds.append(np.sum((positions[0:-i]-positions[i::])**2)/float(num_steps-i))
   return np.array(msds)

def compute_msd(positions, num_steps):
    msds = []
    for t in range(num_steps):
        diff = positions[t] - positions[0]
        msds.append(np.linalg.norm(diff)**2)
    return msds

def msd(trajectory, DT=0.05):
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

def run_for_trajectories(trajectory_b, trajectory_l):
    #msds_b = compute_msd(trajectory_b, STEPS)
    #msds_l = compute_msd(trajectory_l, STEPS)
    msds_b = msd(trajectory_b, 1)
    msds_l = msd(trajectory_l, 1)

    times_b = np.array([i for i in range(1, len(msds_b)+1)])
    times_l = np.array([i for i in range(1, len(msds_l)+1)])

    f, ax = plt.subplots()
    ax.plot(times_b, msds_b, color="deepskyblue")
    ax.scatter(times_b, msds_b, color="purple")
    ax.set_xlabel('Time interval')
    ax.set_ylabel('msd B')
    ax.grid(True)
    plt.show()

    f, ax = plt.subplots()
    ax.plot(times_l, msds_l, color="deepskyblue")
    ax.scatter(times_l, msds_l, color="purple")
    ax.set_xlabel('Time interval')
    ax.set_ylabel('msd L')
    ax.grid(True)
    plt.show()

    print(f"B: {msds_b[-1]}, L: {msds_l[-1]}")

    plt.figure()
    plt.plot(range(1, len(msds_b) + 1), msds_b, color='purple', label='MSD')
    plt.xlabel('Time interval')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Mean Squared Displacement (MSD)')
    plt.title('Mean Squared Displacement (MSD) Analysis')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(range(1, len(msds_l) + 1), msds_l, color='purple', label='MSD')
    plt.xlabel('Time interval')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Mean Squared Displacement (MSD)')
    plt.title('Mean Squared Displacement (MSD) Analysis')
    plt.grid(True)
    plt.legend()
    plt.show()


    popt, pcov = curve_fit(power_law, range(1, len(msds_b) + 1), msds_b)
    alpha = popt[0]
    print(f"Scaling exponent (α) B: {alpha}")
    print(f"MSD at last step B: {msds_b[-1]}")

    popt, pcov = curve_fit(power_law, range(1, len(msds_l) + 1), msds_l)
    alpha = popt[0]
    print(f"Scaling exponent (α) L: {alpha}")
    print(f"MSD at last step L: {msds_l[-1]}")

STEPS = 100000

DIST_TYPE = 3
A = 1.5         
B = 1
#trajectory = levy_walk_simulation(STEPS, DIST_TYPE, A, B)
trajectory_l = levy_walk_3(STEPS, A)
trajectory_b = brownian_motion_2d_without_sigma(STEPS)

run_for_trajectories(trajectory_b=trajectory_b, trajectory_l=trajectory_l)

trajectory_l = levy_walk(STEPS, A)
trajectory_b = brownian_motion_2d(STEPS)

run_for_trajectories(trajectory_b=trajectory_b, trajectory_l=trajectory_l)