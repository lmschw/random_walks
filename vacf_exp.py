import numpy as np
import matplotlib.pyplot as plt

import data_loader

def velocity_autocorrelation(trajectory):
    velocities = np.diff(trajectory)
    n = len(velocities)
    vacf = np.zeros(n-1)
    for tau in range(1, n):
        vacf[tau-1] = np.mean(velocities[:n-tau] * velocities[tau:])
    # Normalize the VACF
    vacf /= vacf[0]
    return vacf

for n in [1, 25, 49, 100]:
    for type in ['brown', 'levy', 'correlated']:
        for i in range(1,4):
            filename = f"{type}_free_{n}_run{i}"
            trajectory = data_loader.load_data(f"c:/Users/lschw/dev/mas-random-walk/mas_random_walk/results/2D/2025-03-20_11-21-14/{filename}.pickle")
            # Compute the VACF
            vacf = velocity_autocorrelation(trajectory)

            # Plot the VACF
            plt.figure(figsize=(8, 6))
            plt.plot(np.arange(1, len(vacf) + 1), vacf)
            plt.xlabel('Time Lag (τ)', fontsize=14)
            plt.ylabel('VACF (C_v(τ))', fontsize=14)
            plt.title(f'Velocity Autocorrelation Function (VACF) - {type} - {i}', fontsize=16)
            plt.grid(True)
            plt.savefig(filename)
            #plt.show()