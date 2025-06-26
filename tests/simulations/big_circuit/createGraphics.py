import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sys
import argparse

# Example data
methods = ['LB Memory', 'WholeCell']



parser = argparse.ArgumentParser()

parser.add_argument("mem_file", type=str, help="Path to file containing lb memory stats")
parser.add_argument("wholecell_file", type=str, help="Path to file contatining lb wholecell stats")
parser.add_argument("-lb", action="store_true", help="Show load balance compute time")
args = parser.parse_args()


memory_file = sys.argv[1]
wholecell_file = sys.argv[2]

df = pd.read_csv(memory_file)
mpi_ranks = df["ranks"].tolist()
memory_A = df["mem"].tolist()
memory_A_max = df["mem_max"].tolist()
memory_A_min = df["mem_min"].tolist()
compute_lb_A = df["compute_lb"].tolist()
cell_creation_A = df["cell_creation"].tolist()
run_time_A = df["run_time"].tolist()

df = pd.read_csv(wholecell_file)
memory_B = df["mem"].tolist()
memory_B_max = df["mem_max"].tolist()
memory_B_min = df["mem_min"].tolist()
compute_lb_B = df["compute_lb"].tolist()
cell_creation_B = df["cell_creation"].tolist()
run_time_B = df["run_time"].tolist()


mem_max = max(memory_A_max + memory_B_max)
compute_lb_max = max(compute_lb_A + compute_lb_B)
creation_max = max(cell_creation_A + cell_creation_B)
run_max = max(run_time_A + run_time_B)

mem_min = min(memory_A_min + memory_B_min)
compute_lb_min = min(compute_lb_A + compute_lb_B)
creation_min = min(cell_creation_A + cell_creation_B)
run_min = min(run_time_A + run_time_B)

mem_max *= 1.05
compute_lb_max *= 1.05
creation_max *= 1.05
run_max *= 1.05

mem_min *= 0.95
compute_lb_min *= 0.95
creation_min *= 0.95
run_min *= 0.95



def compute_yerr(avg, minv, maxv):
    return [np.array(avg) - np.array(minv), np.array(maxv) - np.array(avg)]

memory_A_yerr = compute_yerr(memory_A, memory_A_min, memory_A_max)
memory_B_yerr = compute_yerr(memory_B, memory_B_min, memory_B_max)



rr_color = "red"
memory_color = "blue"


errorbar_style_0 = { 'ecolor': memory_color, 'elinewidth': 2, 'capsize': 8, 'capthick': 2}
errorbar_style_1 = {'ecolor': rr_color, 'elinewidth': 2, 'capsize': 8, 'capthick': 2}

if args.lb:
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    lb_axs = axs[0]
    mem_axs = axs[1]
    creation_axs = axs[2]
    run_axs = axs[3]
else:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    mem_axs = axs[0]
    creation_axs = axs[1]
    run_axs = axs[2]


if args.lb:
    lb_axs.errorbar(mpi_ranks, compute_lb_A, label=methods[0], color=memory_color)
    lb_axs.errorbar(mpi_ranks, compute_lb_B, label=methods[1], color=rr_color)
    lb_axs.set_title('Compute LB')
    lb_axs.set_xlabel("MPI Ranks")
    lb_axs.set_ylabel('Seconds')
    lb_axs.grid(True)
    lb_axs.legend()
    lb_axs.set_ylim(compute_lb_min, compute_lb_max)
    lb_axs.minorticks_on()

mem_axs.errorbar(mpi_ranks, memory_A, yerr=memory_A_yerr, label=methods[0], color=memory_color, **errorbar_style_0)
mem_axs.errorbar(mpi_ranks, memory_B, yerr=memory_B_yerr, label=methods[1], color=rr_color, **errorbar_style_1)
mem_axs.set_title("Memory Usage")
mem_axs.set_xlabel("MPI Ranks")
mem_axs.set_ylabel("Memory per Rank(MB)")
mem_axs.grid(True)
mem_axs.legend()
mem_axs.set_ylim(mem_min, mem_max)
mem_axs.minorticks_on()

# Generation time plot
creation_axs.errorbar(mpi_ranks, cell_creation_A, label=methods[0], color=memory_color)
creation_axs.errorbar(mpi_ranks, cell_creation_B, label=methods[1], color=rr_color)
creation_axs.set_title('Cell creation')
creation_axs.set_xlabel("MPI Ranks")
creation_axs.set_ylabel('Seconds')
creation_axs.grid(True)
creation_axs.legend()
creation_axs.set_ylim(creation_min, creation_max)
creation_axs.minorticks_on()

# Run time 
run_axs.errorbar(mpi_ranks, run_time_A, label=methods[0], color=memory_color)
run_axs.errorbar(mpi_ranks, run_time_B, label=methods[1], color=rr_color)
run_axs.set_title('finished Run')
run_axs.set_xlabel("MPI Ranks")
run_axs.set_ylabel('Seconds')
run_axs.grid(True)
run_axs.legend()
run_axs.set_ylim(run_min, run_max)
run_axs.minorticks_on()


# Layout adjustments
plt.tight_layout()
plt.suptitle('Comparison of Load Balance', fontsize=16, y=1.05)
plt.show()