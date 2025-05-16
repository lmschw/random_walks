
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from random_walk_types.levy_walk import levy_walk_simulation, levy_walk, levy_walk_2
from random_walk_types.brownian_motion import brownian_motion_2d_without_sigma
from random_walk_types.correlated_random_walk import correlated_random_walk_2d

def velocity_autocorrelation(trajectory, windows=[1, 2, 3, 4]):
    velocities = calculate_velocity(positions=trajectory)
    vals = []
    for window in windows:
        window_vals = []
        for t in range(0, len(trajectory-window), window):
            window_vals.append(np.sum(velocities[t] * velocities[t+window], axis=1))
        vals.append(np.average(np.sum(window_vals)))
    return vals


def calculate_velocity(positions):
    """Calculate velocity vectors from positions."""
    velocities = np.diff(positions, axis=0)
    return velocities

def autocorrelation(x):
    """Compute the autocorrelation of a signal."""
    n = len(x)
    variance = np.var(x)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode='full')[-n:]
    result = r / (variance * n)
    return result

def power_law(x, a):
    return np.power(x, a/2)

def log_power_law(x, a):
    return np.log(power_law(x, a))

def log_power_abs_diff(xy, a):
    x = xy[0]
    vacf = xy[1]
    return np.sqrt(np.abs(power_law(x, a)-vacf))

def plot_vacf_msd_and_boxplot(trajectory, motion_type):
    """Plot Velocity Autocorrelation Function (VACF), Mean Squared Displacement (MSD), and a boxplot of VACF."""
    velocities = calculate_velocity(trajectory)
    # vacf = (autocorrelation(velocities[:,0]) + autocorrelation(velocities[:,1]))/2
    vacf = (autocorrelation(velocities[:, 0]) + autocorrelation(velocities[:, 1]))/2
    vacf = vacf[:10000]

    lags = np.arange(1, len(vacf) + 1)
    params, _ = curve_fit(log_power_abs_diff, (lags, vacf), np.zeros_like(vacf), p0=[0.8])
    print(f"Params: {params}")
    fitted_vacf = power_law(lags, *params)

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))

    # Plot VACF
    axs.plot(vacf, 'o', label="VACF", color='k', markeredgecolor='k', fillstyle='none')
    axs.plot(lags, fitted_vacf, 'r--', label=f'Fitted Curve a={params[0]:.2f}')
    axs.set_xscale('log')
    # axs.set_yscale('log')
    axs.set_title(f'Velocity Autocorrelation Function for {motion_type}')
    axs.set_xlabel('Lag')
    axs.set_ylabel('Autocorrelation')
    # axs.set_ylim(0, 1)
    axs.grid(True)

    plt.tight_layout()
    plt.show()

STEPS = 1000000

DIST_TYPE = 3
A = 1.5         
B = 1
#trajectory_l = levy_walk_simulation(STEPS, DIST_TYPE, A, B)
trajectory_l = levy_walk(STEPS, A)
trajectory_b = brownian_motion_2d_without_sigma(STEPS)
trajectory_c = correlated_random_walk_2d(STEPS)

plot_vacf_msd_and_boxplot(trajectory=trajectory_l, motion_type="Lévy walk")

plot_vacf_msd_and_boxplot(trajectory=trajectory_b, motion_type="Brownian motion")

plot_vacf_msd_and_boxplot(trajectory=trajectory_c, motion_type="CRW")