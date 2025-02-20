
import numpy as np
import matplotlib.pyplot as plt

from levy_walk import levy_walk_simulation
from levy_walk_2 import levy_walk
from levy_walk_3 import levy_walk_3
from brownian_motion import brownian_motion_2d, brownian_motion_2d_without_sigma

def velocity_autocorrelation(trajectory):
    """
    Compute the velocity autocorrelation function for a given trajectory.
    trajectory: numpy array of positions at discrete time steps.
    Returns: numpy array of the velocity autocorrelation function.
    """
    # Compute velocities
    velocities = np.diff(trajectory)
    
    # Compute VACF for different time lags
    n = len(velocities)
    vacf = np.zeros(n-1)
    
    for tau in range(1, n):
        vacf[tau-1] = np.mean(velocities[:n-tau] * velocities[tau:])
    
    # Normalize the VACF
    vacf /= vacf[0]
    
    return vacf

STEPS = 100000

DIST_TYPE = 3
A = 1.5         
B = 1
#trajectory_l = levy_walk_simulation(STEPS, DIST_TYPE, A, B)
trajectory_l = levy_walk_3(STEPS, A)
trajectory_b = brownian_motion_2d_without_sigma(STEPS)


# Compute the VACF
vacf = velocity_autocorrelation(trajectory_b)

# Plot the VACF
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(vacf) + 1), vacf)
plt.xlabel('Time Lag (τ)', fontsize=14)
plt.ylabel('VACF (C_v(τ))', fontsize=14)
plt.title('Velocity Autocorrelation Function (VACF) - Brownian motion', fontsize=16)
plt.grid(True)
plt.show()

# Compute the VACF
vacf = velocity_autocorrelation(trajectory_l)

# Plot the VACF
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(vacf) + 1), vacf)
plt.xlabel('Time Lag (τ)', fontsize=14)
plt.ylabel('VACF (C_v(τ))', fontsize=14)
plt.title('Velocity Autocorrelation Function (VACF) - Lévy walk', fontsize=16)
plt.grid(True)
plt.show()