import numpy as np
import matplotlib.pyplot as plt
from levy_walk import levy_walk_simulation
#from levy_walk_2 import levy_walk
from levy_walk_3 import levy_walk_3
from brownian_motion import brownian_motion_2d_without_sigma

N = 1000000
A = 1.5

bm_data = brownian_motion_2d_without_sigma(N)[:,0] # just 1d
levy_data = levy_walk_3(N, A)[:,0] # just 1d

# Plotting the data
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(bm_data, label='Brownian Motion')
plt.title('Brownian Motion')
plt.subplot(2, 1, 2)
plt.plot(levy_data, label='Levy Walk', color='r')
plt.title('Levy Walk')
plt.tight_layout()
plt.show()

from scipy.stats import entropy

def diffusion_entropy(trajectory, num_bins=50):
    displacements = np.diff(trajectory)
    hist, bin_edges = np.histogram(displacements, bins=num_bins, density=True)
    return entropy(hist + 1e-10)  # Add small value to avoid log(0)

def diffusion_entropy_log_bins(trajectory, num_bins=50):
    displacements = np.diff(trajectory)
    log_bins = np.logspace(np.log10(np.min(displacements[displacements > 0])), 
                           np.log10(np.max(displacements)), num_bins)
    hist, _ = np.histogram(displacements, bins=log_bins, density=True)
    return entropy(hist + 1e-10)  # Add a small constant to avoid log(0)

def diffusion_entropy_absolute(trajectory, num_bins=50):
    displacements = np.abs(np.diff(trajectory))  # Use absolute displacements
    log_bins = np.logspace(np.log10(np.min(displacements[displacements > 0])), 
                np.log10(np.max(displacements)), num_bins)
    hist, _ = np.histogram(displacements, bins=log_bins, density=True)
    return entropy(hist + 1e-10)  # Add small constant to avoid log(0)

def diffusion_entropy_standardized(trajectory, num_bins=50):
    displacements = np.diff(trajectory)
    # Standardize displacements (mean 0, std 1)
    displacements = (displacements - np.mean(displacements)) / np.std(displacements)
    log_bins = np.logspace(np.log10(np.min(displacements[displacements > 0])), 
                        np.log10(np.max(displacements)), num_bins)

    hist, _ = np.histogram(displacements, bins=log_bins, density=True)
    return entropy(hist + 1e-10)

def shannon_entropy(distribution):
    distribution = distribution / np.sum(distribution)
    return -np.sum(distribution * np.log(distribution + 1e-10))  # Add small value to avoid log(0)

def diffusion_entropy_shannon(trajectory, num_bins=50):
    displacements = np.diff(trajectory)
    displacements = (displacements - np.mean(displacements)) / np.std(displacements)
    log_bins = np.logspace(np.log10(np.min(displacements[displacements > 0])), 
                    np.log10(np.max(displacements)), num_bins)

    hist, _ = np.histogram(displacements, bins=log_bins, density=True)
    return shannon_entropy(hist)

entropy_bm = diffusion_entropy(bm_data)
entropy_levy = diffusion_entropy(levy_data)

print(f'Diffusion Entropy for Brownian Motion: {entropy_bm}')
print(f'Diffusion Entropy for Lévy Walk: {entropy_levy}')

entropy_bm_log = diffusion_entropy_log_bins(bm_data)
entropy_levy_log = diffusion_entropy_log_bins(levy_data)

print(f"Diffusion Entropy (log bins) for Brownian Motion: {entropy_bm_log}")
print(f"Diffusion Entropy (log bins) for Lévy Walk: {entropy_levy_log}")

entropy_bm_standardized = diffusion_entropy_standardized(bm_data)
entropy_levy_standardized = diffusion_entropy_standardized(levy_data)

print(f"Standardized Diffusion Entropy for Brownian Motion: {entropy_bm_standardized}")
print(f"Standardized Diffusion Entropy for Lévy Walk: {entropy_levy_standardized}")

entropy_bm_absolute = diffusion_entropy_absolute(bm_data)
entropy_levy_absolute = diffusion_entropy_absolute(levy_data)

print(f"Absolute Diffusion Entropy for Brownian Motion: {entropy_bm_absolute}")
print(f"Absolute Diffusion Entropy for Lévy Walk: {entropy_levy_absolute}")

entropy_bm_shannon = diffusion_entropy_shannon(bm_data)
entropy_levy_shannon = diffusion_entropy_shannon(levy_data)

print(f"Shannon Diffusion Entropy for Brownian Motion: {entropy_bm_shannon}")
print(f"Shannon Diffusion Entropy for Lévy Walk: {entropy_levy_shannon}")