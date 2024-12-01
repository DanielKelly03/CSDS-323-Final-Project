import matplotlib.pyplot as plt
import numpy as np

# Data for the methods (replacing None with 0 or placeholder)
methods = ["SVD", "SLR", "SLR-w", "SLR-a", "SLR-c"]
compression_ratios_r6 = [0.00125, 0.00075, 0, 0, 0]  # Replacing None with 0
compression_ratios_r64 = [0.0021, 0.0045, 0, 0, 0]
accuracies_r6 = [0.382, 0, 0.308, 0.280, 0.380]  # Replacing None with 0
accuracies_r64 = [0.434, 0, 0.436, 0.436, 0.436]

# Plotting the bar graph for both r=6 and r=64
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Bar plot for Compression Ratio (r=6 and r=64)
bar_width = 0.35
index = np.arange(len(methods))

bar1 = ax[0].bar(index, compression_ratios_r6, bar_width, label='Rank r=6', color='blue')
bar2 = ax[0].bar(index + bar_width, compression_ratios_r64, bar_width, label='Rank r=64', color='orange')

# Adding labels and title for Compression Ratio
ax[0].set_xlabel('Methods')
ax[0].set_ylabel('Compression Ratio')
ax[0].set_title('Compression Ratio by Method and Rank')

# Only label the first two methods for the Compression Ratio graph
ax[0].set_xticks(index[:2] + bar_width / 2)  # Labels only for the first two methods
ax[0].set_xticklabels(methods[:2])  # Labels for SVD and SLR only
ax[0].legend()

# Bar plot for Accuracy (r=6 and r=64)
bar3 = ax[1].bar(index, accuracies_r6, bar_width, label='Rank r=6', color='blue')
bar4 = ax[1].bar(index + bar_width, accuracies_r64, bar_width, label='Rank r=64', color='orange')

# Adding labels and title for Accuracy
ax[1].set_xlabel('Methods')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Accuracy by Method and Rank')
ax[1].set_xticks(index + bar_width / 2)
ax[1].set_xticklabels(methods)  # Keep all labels for the Accuracy graph
ax[1].legend()

plt.tight_layout()
plt.show()
