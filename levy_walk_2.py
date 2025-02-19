import numpy as np
import matplotlib.pyplot as plt

def levy_walk(num_steps, alpha=1.5):
    # Generate Lévy-distributed step sizes using a power-law distribution
    # Lévy distribution has the form of p(x) ~ x^(-1-α)
    step_sizes = np.random.standard_t(df=alpha, size=num_steps)
    
    # Initialize the starting position
    x, y = 0, 0
    positions = [(x, y)]
    
    for step in step_sizes:
        # Choose a random angle for the direction (uniform distribution between 0 and 2π)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Update position
        x += step * np.cos(angle)
        y += step * np.sin(angle)
        
        positions.append((x, y))
    
    return np.array(positions)

if __name__ == "__main__":
    # Parameters
    num_steps = 1000
    alpha = 1.5  # Lévy index, often between 1.5 and 2

    # Perform Lévy walk
    positions = levy_walk(num_steps, alpha)

    # Plot the path
    plt.plot(positions[:, 0], positions[:, 1])
    plt.title(f'Lévy Walk with {num_steps} steps and α = {alpha}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    plt.figure(figsize=(12, 6))

    pdf = ""
    # Trajectory
    plt.subplot(1, 2, 1)
    plt.plot(positions[:, 0], positions[:, 1], marker='o', linestyle='-', color='b')
    plt.title( pdf + ' Trajectory')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)

    # Pdf
    plt.subplot(1, 2, 2)
    step_sizes = np.linalg.norm(positions[1:] - positions[:-1], axis=1)  
    plt.hist(step_sizes, bins=20, density=True, alpha=0.7, color='green', edgecolor='black')
    plt.title('Probability Density Distribution of Step Sizes')
    plt.xlabel('Step Size')
    plt.ylabel('Probability Density')
    plt.grid(True)

    plt.tight_layout()
    plt.show()