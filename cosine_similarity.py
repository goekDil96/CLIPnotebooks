import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

# Define three sets of vectors with different angles
vectors = [
    (np.array([3, 4]), np.array([1.5, 2]), np.array([0.0000000001, 0])),
    (np.array([3, 4]), np.array([3, 1]), np.array([0.0000000001, 0])),
    (np.array([3, 0]), np.array([0, 4]), np.array([0.0000000001, 0.00000000000000000000001]))
]

# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (A, B, C) in zip(axes, vectors):
    # Calculate cosine similarity and angle
    cosine_similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    cos_sim_2 = np.dot(C, B) / (np.linalg.norm(C) * np.linalg.norm(B))
    angle_rad = np.arccos(cosine_similarity)
    angle_deg = np.degrees(angle_rad)

    # Plot vectors
    ax.quiver(0, 0, A[0], A[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.02)
    ax.quiver(0, 0, B[0], B[1], angles='xy', scale_units='xy', scale=1, color='b', width=0.02)

    # Set limits and labels
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Angle between vectors θ = {angle_rad:.2f}\ncos(θ) = {cosine_similarity:.2f}')

    # Add grid
    ax.grid(True)
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()
