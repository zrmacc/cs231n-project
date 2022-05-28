"""
Plot sections.

@author: zmccaw
"""

import matplotlib.pyplot as plt
import numpy as np

pos = np.load("data/pos_d1/pos_0.npy")
pos = pos * 255.0

neg = np.load("data/neg_d1/neg_0.npy")
neg = neg * 255.0

# Plot.
fig, axes = plt.subplots(3, 2, figsize=(5, 8))
# fig.suptitle('Orthogonal sections')
axes[0, 0].set_title("Positive example")
axes[0, 1].set_title("Negative example")
row_labs = ["XY plane", "YZ plane", "XZ plane"]

for i in range(3):
    axes[i, 0].imshow(pos[:, :, i], interpolation='nearest')
    axes[i, 1].imshow(neg[:, :, i], interpolation='nearest')
    
    for j in range(2):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
    
    for ax, row in zip(axes[:, 0], row_labs):
        ax.set_ylabel(row)

fig.savefig("sections.png", dpi=480)
