import pickle
import numpy as np
import matplotlib.pyplot as plt
"""
Visualizes a trajectory from a pickle file.
"""

filename = "c:/Users/Lilly/dev/mas-random-walk/mas_random_walk/results/2D/2025-05-15_16-34-18/old_brown_free_25_run1.pickle"
file = open(filename,'rb')
object_file = pickle.load(file)
trajectory = np.array(object_file["trajectory"])
print()

plt.plot(trajectory[:,0], trajectory[:,1])
plt.title(f'Test')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()