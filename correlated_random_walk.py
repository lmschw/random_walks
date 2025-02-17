import numpy as np
import matplotlib.pyplot as plt

def correlated_random_walk_2d(num_steps, rho=0.8, sigma=0.1):
    # Initialize positions and angles
    positions_x = np.zeros(num_steps)
    positions_y = np.zeros(num_steps)
    angles_x = np.zeros(num_steps)
    angles_y = np.zeros(num_steps)
    
    # Start with random angles for the first step in both x and y directions
    angles_x[0] = np.random.uniform(0, 2 * np.pi)
    angles_y[0] = np.random.uniform(0, 2 * np.pi)
    
    # Generate the random walk
    for i in range(1, num_steps):
        # Update the angles using the correlation parameter rho
        noise_x = np.random.normal(0, sigma)  # Gaussian noise for x direction
        noise_y = np.random.normal(0, sigma)  # Gaussian noise for y direction
        angles_x[i] = rho * angles_x[i-1] + (1 - rho) * noise_x  # Correlated angle for x
        angles_y[i] = rho * angles_y[i-1] + (1 - rho) * noise_y  # Correlated angle for y
        
        # Update positions in 2D using the cosine and sine of the angles
        positions_x[i] = positions_x[i-1] + np.cos(angles_x[i])
        positions_y[i] = positions_y[i-1] + np.sin(angles_y[i])
    
    return positions_x, positions_y

# Parameters
num_steps = 1000
rho = 0.8  # Correlation parameter, controls persistence
sigma = 0.1  # Noise strength

# Perform 2D correlated random walk
positions_x, positions_y = correlated_random_walk_2d(num_steps, rho, sigma)

# Plot the correlated random walk path
plt.plot(positions_x, positions_y)
plt.title(f'2D Correlated Random Walk with {num_steps} steps, ρ = {rho}')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()
