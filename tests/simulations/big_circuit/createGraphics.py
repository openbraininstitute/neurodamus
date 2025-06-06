import matplotlib.pyplot as plt

import numpy as np

# Example data
methods = ['LB Memory', 'Round Robin']
memory_usage = [495.59, 489.35]       # in MB
memory_min = [480.67, 311.78]
memory_max = [504.91, 738.22]
memory_yerr = [
    [mem - minv for mem, minv in zip(memory_usage, memory_min)],
    [maxv - mem for mem, maxv in zip(memory_usage, memory_max)]
]

generation_time = [35.95, 58.47]    # in seconds
run_time = [25.88, 43.93]           # in seconds

x = np.arange(len(methods))  # [0, 1]
width = 0.6

# Create 3 subplots (1 row, 3 columns)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

errorbar_style = {
    'ecolor': 'black',
    'elinewidth': 2,
    'capsize': 8,
    'capthick': 2
}


# Memory usage plot
axs[0].bar(x, memory_usage, yerr=memory_yerr, color=['skyblue', 'lightgreen'], width=width, error_kw=errorbar_style)
axs[0].set_title('Memory Usage')
axs[0].set_ylabel('MB')
axs[0].set_xticks(x)
axs[0].set_xticklabels(methods)

axs[0].grid(True, axis='y', linestyle='--', linewidth=0.5)
axs[0].minorticks_on()


# Generation time plot
axs[1].bar(x, generation_time, color=['skyblue', 'lightgreen'], width=width)
axs[1].set_title('Cell creation')
axs[1].set_ylabel('Seconds')
axs[1].set_xticks(x)
axs[1].set_xticklabels(methods)

axs[1].grid(True, axis='y', linestyle='--', linewidth=0.5)
axs[1].minorticks_on()
axs[1].set_ylim(0, 60)  # Generation Time y-axis from 0 to 100

# Run time plot
axs[2].bar(x, run_time, color=['skyblue', 'lightgreen'], width=width)
axs[2].set_title('finished Run')
axs[2].set_ylabel('Seconds')
axs[2].set_xticks(x)
axs[2].set_xticklabels(methods)

axs[2].grid(True, axis='y', linewidth=0.5)
axs[2].minorticks_on()
axs[2].set_ylim(0, 60) 

# Layout adjustments
plt.tight_layout()
plt.suptitle('Comparison of Load Balance', fontsize=16, y=1.05)
plt.show()