import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sys
import argparse

# Example data
methods = []



parser = argparse.ArgumentParser()

parser.add_argument("input_files", type=str, nargs='+', help="Paths to one or more stats files")
parser.add_argument("-lb", action="store_true", help="Show load balance compute time")
args = parser.parse_args()


memory = []
memory_max = []
memory_min = []
memory_yerr = []
compute_lb = []
cell_creation = []
run_time = []

def compute_yerr(avg, minv, maxv):
    return [np.array(avg) - np.array(minv), np.array(maxv) - np.array(avg)]

methods = [f.split("_")[1].split(".")[0] for f in args.input_files]
for file in args.input_files:
    df = pd.read_csv(file)
    mpi_ranks = df["ranks"].tolist()
    mem = df["mem"].tolist()
    memory.append(mem)
    mmax = df["mem_max"].tolist()
    memory_max.append(mmax)
    mmin = df["mem_min"].tolist()
    memory_min.append(mmin)
    compute_lb.append(df["compute_lb"].tolist())
    cell_creation.append(df["cell_creation"].tolist())
    run_time.append(df["run_time"].tolist())
    memory_yerr.append(compute_yerr(mem, mmin, mmax))

mem_max = max([max(v) for v in memory_max]) * 1.05
mem_min = min([min(v) for v in memory_min]) * 0.95
compute_lb_max = max([max(v) for v in compute_lb]) * 1.05 
compute_lb_min = min([min(v) for v in compute_lb]) * 0.95
creation_max = max([max(v) for v in cell_creation]) * 1.05 
creation_min = min([min(v) for v in cell_creation]) * 0.95 
run_max = max([max(v) for v in run_time]) * 1.05 
run_min = min([min(v) for v in run_time]) * 0.95


colors = plt.get_cmap("tab10")
method_colors = [colors(i) for i, method in enumerate(methods)]


errorbar_style = []
for i in range(len(methods)):
    errorbar_style.append({ 'ecolor': method_colors[i], 'elinewidth': 2, 'capsize': 8, 'capthick': 2})

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
    for i in range(len(methods)):
        lb_axs.errorbar(mpi_ranks, compute_lb[i], label=methods[i], color=method_colors[i])
    lb_axs.set_title('Compute LB')
    lb_axs.set_xlabel("MPI Ranks")
    lb_axs.set_ylabel('Seconds')
    lb_axs.grid(True)
    lb_axs.legend()
    lb_axs.set_ylim(compute_lb_min, compute_lb_max)
    lb_axs.minorticks_on()

for i in range(len(methods)):
    mem_axs.errorbar(mpi_ranks, memory[i], yerr=memory_yerr[i], label=methods[i], color=method_colors[i], **errorbar_style[i])
mem_axs.set_title("Memory Usage")
mem_axs.set_xlabel("MPI Ranks")
mem_axs.set_ylabel("Memory per Rank(MB)")
mem_axs.grid(True)
mem_axs.legend()
mem_axs.set_ylim(mem_min, mem_max)
mem_axs.minorticks_on()

# Generation time plot
for i in range(len(methods)):
    creation_axs.errorbar(mpi_ranks, cell_creation[i], label=methods[i], color=method_colors[i])
creation_axs.set_title('Cell creation')
creation_axs.set_xlabel("MPI Ranks")
creation_axs.set_ylabel('Seconds')
creation_axs.grid(True)
creation_axs.legend()
creation_axs.set_ylim(creation_min, creation_max)
creation_axs.minorticks_on()

# Run time 
for i in range(len(methods)):
    run_axs.errorbar(mpi_ranks, run_time[i], label=methods[i], color=method_colors[i])
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