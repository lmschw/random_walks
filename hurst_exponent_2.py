import numpy as np
import matplotlib.pyplot as plt

# Generate a 2D random walk
def generate_random_walk(n_steps):
    # Random walk steps in x and y directions
    x_steps = np.random.choice([-1, 1], size=n_steps)
    y_steps = np.random.choice([-1, 1], size=n_steps)
    
    # Cumulative sum to get the positions
    x_positions = np.cumsum(x_steps)
    y_positions = np.cumsum(y_steps)
    
    return x_positions, y_positions

# Rescaled Range (R/S) analysis for a given data
def hurst_exponent(data):
    N = len(data)
    mean_data = np.mean(data)
    Y = np.cumsum(data - mean_data)  # Cumulative sum of deviations
    
    # Calculate rescaled range R/S
    R = np.max(Y) - np.min(Y)
    S = np.std(data)
    
    return R / S

# Calculate the Hurst exponent using R/S method
def calculate_hurst(x_positions, y_positions):
    # Apply Hurst exponent calculation to both x and y coordinates
    Hx = hurst_exponent(x_positions)
    Hy = hurst_exponent(y_positions)
    
    # Combine the results by averaging the exponents
    H_combined = (Hx + Hy) / 2
    return Hx, Hy, H_combined

# Main code for generating the random walk and calculating the Hurst exponent
n_steps = 1000  # Number of steps in the random walk

# Generate the random walk for x and y
x_positions, y_positions = generate_random_walk(n_steps)

# Calculate the Hurst exponents
Hx, Hy, H_combined = calculate_hurst(x_positions, y_positions)

# Display the results
print(f"Hurst exponent for x-coordinates: {Hx}")
print(f"Hurst exponent for y-coordinates: {Hy}")
print(f"Combined Hurst exponent: {H_combined}")

# Optionally, plot the random walk in 2D for visualization
plt.figure(figsize=(8, 8))
plt.plot(x_positions, y_positions, label="Random Walk Path")
plt.title("2D Random Walk")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.show()