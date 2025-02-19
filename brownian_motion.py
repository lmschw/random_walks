import numpy as np
import matplotlib.pyplot as plt

def brownian_motion_2d(num_steps, dt=1.0, sigma=1.0):
    # Generate independent steps for x and y directions
    steps_x = np.random.normal(0, sigma * np.sqrt(dt), num_steps)
    steps_y = np.random.normal(0, sigma * np.sqrt(dt), num_steps)
    
    # Cumulative sum for x and y to get the position at each step
    positions_x = np.cumsum(steps_x)
    positions_y = np.cumsum(steps_y)
    
    return np.column_stack((positions_x, positions_y))

if __name__ == "__main__":
    # Parameters
    num_steps = 1000
    dt = 1.0  # Time step
    sigma = 1.0  # Standard deviation of the step

    # Perform 2D Brownian motion
    positions = brownian_motion_2d(num_steps, dt, sigma)

    # Plot the 2D Brownian motion path
    plt.plot(positions[:,0], positions[:,1])
    plt.title(f'2D Brownian Motion with {num_steps} steps')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()