import matplotlib.pyplot as plt

import numpy as np

# Example data
methods = ['LB Memory', 'Round Robin']

mpi_ranks =         [2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12,     13,     14]
memory_A =          [702.99, 503.84, 403.10, 356.02, 303.52, 273.58, 255.38, 257.84, 257.18, 238.93, 213.42, 203.56, 198.43]
memory_A_min =      [695.45, 488.83, 381.50, 344.16, 288.81, 255.78, 248.09, 250.91, 247.06, 224.77, 196.44, 196.12, 192.77] 
memory_A_max =      [710.53, 511.91, 433.02, 370.25, 338.12, 284.91, 261.30, 271.42, 263.73, 248.56, 223.77, 212.59, 205.39]
generation_time_A = [57.79,  39.93,  30.96,  24.96,  20.78,  18.74,  16.92,  15.73,  14.39,  13.57,  12.98,  12.32,  12.54] 
run_time_A =        [41.30,  26.26,  19.64,  16.60,  13.43,  9.95,   8.78,   8.70,   7.88,   7.24,   6.19,   6.12,   5.76] 
#mpi_ranks =        [2,      3,      4,      5,      6,      7,      8,      9,      10,     11,     12,     13,     14]
memory_B =          [706.82, 515.30, 400.98, 357.66, 296.00, 281.80, 264.92, 240.77, 246.55, 237.74, 211.19, 204.42, 196.53] 
memory_B_min =      [670.69, 365.91, 382.48, 349.23, 109.39, 272.72, 251.95, 183.44, 229.27, 218.77, 101.48, 192.11, 180.88] 
memory_B_max =      [742.95, 616.62, 422.80, 363.41, 537.12, 295.47, 276.55, 282.53, 261.56, 247.20, 354.59, 212.23, 208.80] 
generation_time_B = [68.47,  48.57,  35.53,  24.07,  46.63,  17.83,  19.01,  18.04,  17.70,  13.93,  30.30,  13.22,  15.13]
run_time_B =        [48.91,  42.26,  20.81,  16.13,  35.63,  10.37,  9.48,   12.02,  8.97,   7.90,   18.16,  6.25,   6.30]

def compute_yerr(avg, minv, maxv):
    return [np.array(avg) - np.array(minv), np.array(maxv) - np.array(avg)]

memory_A_yerr = compute_yerr(memory_A, memory_A_min, memory_A_max)
memory_B_yerr = compute_yerr(memory_B, memory_B_min, memory_B_max)


# Create 3 subplots (1 row, 3 columns)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
rr_color = "red"
memory_color = "blue"


errorbar_style_0 = { 'ecolor': memory_color, 'elinewidth': 2, 'capsize': 8, 'capthick': 2}

errorbar_style_1 = {'ecolor': rr_color, 'elinewidth': 2, 'capsize': 8, 'capthick': 2}

axs[0].errorbar(mpi_ranks, memory_A, yerr=memory_A_yerr, label=methods[0], color=memory_color, **errorbar_style_0)
axs[0].errorbar(mpi_ranks, memory_B, yerr=memory_B_yerr, label=methods[1], color=rr_color, **errorbar_style_1)
axs[0].set_title("Memory Usage")
axs[0].set_xlabel("MPI Ranks")
axs[0].set_ylabel("Memory per Rank(MB)")
axs[0].grid(True)
axs[0].legend()
axs[0].set_ylim(0, 750)
axs[0].minorticks_on()

# Generation time plot
axs[1].errorbar(mpi_ranks, generation_time_A, label=methods[0], color=memory_color)
axs[1].errorbar(mpi_ranks, generation_time_B, label=methods[1], color=rr_color)
axs[1].set_title('Cell creation')
axs[1].set_xlabel("MPI Ranks")
axs[1].set_ylabel('Seconds')
axs[1].grid(True)
axs[1].legend()
axs[1].set_ylim(0, 70)
axs[1].minorticks_on()

# Run time 
axs[2].errorbar(mpi_ranks, run_time_A, label=methods[0], color=memory_color)
axs[2].errorbar(mpi_ranks, run_time_B, label=methods[1], color=rr_color)
axs[2].set_title('finished Run')
axs[2].set_xlabel("MPI Ranks")
axs[2].set_ylabel('Seconds')
axs[2].grid(True)
axs[2].legend()
axs[2].set_ylim(0, 70)
axs[2].minorticks_on()


# Layout adjustments
plt.tight_layout()
plt.suptitle('Comparison of Load Balance', fontsize=16, y=1.05)
plt.show()