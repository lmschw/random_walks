import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from levy_walk import levy_walk_simulation
from levy_walk_2 import levy_walk
from levy_walk_3 import levy_walk_3
from brownian_motion import brownian_motion_2d, brownian_motion_2d_without_sigma


def calculate_velocity(positions):
    """Calculate velocity vectors from positions."""
    # print(positions)
    velocities = np.diff(positions, axis=0)
    # d_r = np.hypot(velocities[:, 0], velocities[:, 1])

    # print(f"d_r: {np.min(d_r), np.max(d_r)}")
    # d_r = d_r/np.max(d_r)
    # d_theta = (np.arctan2(positions[:, 1], positions[:, 0]) + np.pi)/(2*np.pi)
    # print(f"d_theta: {d_theta}")

    # velocities = np.vstack((d_r, d_theta)).T
    # print(f"Vel: {velocities}")
    # print(f"Vel: {np.min(velocities, axis=0)}")
    return velocities

# def autocorrelation(x):
#     """Compute the autocorrelation of a signal."""
#     n = len(x)
#     variance = np.var(x)
#     x = x - np.mean(x)
#     r = np.correlate(x, x, mode='full')[-n:]
#     result = r / (variance * n)
#     return result

def autocorrelation(trajectory):
    vacfs = []
    for t in range(len(trajectory)):
        scalar_products = np.dot(trajectory[0], trajectory[t])
        vacfs.append(np.average(scalar_products))
    return vacfs

def calculate_diffusion_coefficient_vacf(vacfs):
    return (np.trapz(vacfs))/3

def calculate_msd(trajectory):
    """Calculate Mean Squared Displacement (MSD) from trajectory."""
    msd = np.zeros(len(trajectory))
    for i in range(len(trajectory)):
        differences = trajectory[i:] - trajectory[:-i] if i else trajectory
        squared_displacements = np.square(differences).sum(axis=1)
        msd[i] = np.mean(squared_displacements)

    return msd

def power_law(x, a):
    return np.power(x, a/2)

def log_power_law(x, a):
    return np.log(power_law(x, a))

def log_power_abs_diff(xy, a):
    x = xy[0]
    vacf = xy[1]
    return np.sqrt(np.abs(power_law(x, a)-vacf))

def plot_vacf_msd_and_boxplot(trajectory):
    """Plot Velocity Autocorrelation Function (VACF), Mean Squared Displacement (MSD), and a boxplot of VACF."""
    velocities = calculate_velocity(trajectory)
    print(velocities)
    # vacf = (autocorrelation(velocities[:,0]) + autocorrelation(velocities[:,1]))/2
    vacf = autocorrelation(trajectory)
    #vacf = vacf[:10000]

    lags = np.arange(1, len(vacf) + 1)
    params, _ = curve_fit(log_power_abs_diff, (lags, vacf), np.zeros_like(vacf), p0=[0.8])
    print(f"Params: {params}")
    fitted_vacf = power_law(lags, *params)


    msd = calculate_msd(trajectory)

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    print(f"D_vacf = {calculate_diffusion_coefficient_vacf(vacf)}")

    # Plot VACF
    axs.plot(vacf, 'o', label="VACF", color='k', markeredgecolor='k', fillstyle='none')
    axs.plot(lags, fitted_vacf, 'r--', label=f'Fitted Curve a={params[0]:.2f}')
    axs.set_xscale('log')
    # axs.set_yscale('log')
    axs.set_title('Velocity Autocorrelation Function')
    axs.set_xlabel('Lag')
    axs.set_ylabel('Autocorrelation')
    # axs.set_ylim(0, 1)
    axs.grid(True)


    # # Plot MSD
    # axs[1].plot(msd, label="MSD")
    # axs[1].set_title('Mean Squared Displacement')
    # axs[1].set_xlabel('Time step')
    # axs[1].set_xscale('log')
    # # axs[1].set_yscale('log')
    # axs[1].set_ylabel('MSD')
    # axs[1].grid(True)

    # # Angle distribution plot
    # axs[2].hist(velocities, bins=50, color='gray', edgecolor='black')
    # axs[2].set_title('Angle Distribution')
    # axs[2].set_xlabel('Angle (radians)')
    # axs[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

STEPS = 100000

DIST_TYPE = 3
A = 1.5         
B = 1
#trajectory_l = levy_walk_simulation(STEPS, DIST_TYPE, A, B)
trajectory_l = levy_walk_3(STEPS, A)
trajectory_b = brownian_motion_2d_without_sigma(STEPS)
plot_vacf_msd_and_boxplot(trajectory_b)
plot_vacf_msd_and_boxplot(trajectory_l)
