import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import utils.data_loader as data_loader

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

def run_for_trajectory(trajectory, name):
    msds = msd(trajectory, 1)
    times = np.array([i for i in range(1, len(msds)+1)])

    print(f"{name}: {msds[-1]}")

    f, ax = plt.subplots()
    ax.plot(times, msds, color="deepskyblue")
    ax.scatter(times, msds, color="purple")
    ax.set_xlabel('Time interval')
    ax.set_ylabel(f'msd {name}')
    ax.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(1, len(msds) + 1), msds, color='purple', label='MSD')
    plt.xlabel('Time interval')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Mean Squared Displacement (MSD)')
    plt.title(f'Mean Squared Displacement (MSD) Analysis {name}')
    plt.grid(True)
    plt.legend()
    plt.show()

    popt, pcov = curve_fit(power_law, range(1, len(msds) + 1), msds)
    alpha = popt[0]
    print(f"Scaling exponent (α) {name}: {alpha}")
    print(f"MSD at last step {name}: {msds[-1]}")



for n in [1, 25, 49, 100]:
    for type in ['brown']:
        for i in range(1,4):
            filename = f"{type}_free_{n}_run{i}"
            trajectory = data_loader.load_data(f"c:/Users/lschw/dev/mas-random-walk/mas_random_walk/results/2D/2025-03-20_11-21-14/{filename}.pickle")
            run_for_trajectory(trajectory=trajectory, name=f'{type} - {n} - {i}')
