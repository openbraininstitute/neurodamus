from pathlib import Path
import sys
import subprocess
import shutil
import re

import argparse

def get_mode(args):
    mode = "--lb-mode="
    if len(args) < 2:
        name = "memory"
    else:
        name = args[1]
    mode += name 
    print("Mode ", mode)
    return mode, name 

def clean_files(paths, expresions = []):

    for p in paths:
        p = Path(p)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()

    for file in Path().iterdir():
        for exp in expresions:
            if exp in file.name:
                if file.is_dir():
                    shutil.rmtree(file)
                elif file.is_file():
                    file.unlink()

def data_to_csv(file_name, num_ranks, mem, mem_max, mem_min, compute_lb, cell_creation, run_time):
    import csv
    header = [ "ranks", "mem", "mem_max", "mem_min", "compute_lb", "cell_creation", "run_time"]
    rows = zip(num_ranks, mem, mem_max, mem_min, compute_lb, cell_creation, run_time)

    with open(file_name, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-mode", type=str, default="memory", help="loab balance mode")
    parser.add_argument("-cache", action="store_true", help="Use cached data if available")
    parser.add_argument("-log", action="store_true", help="Keep log files")

    args = parser.parse_args()
    mode = "--lb-mode=" + args.mode

    print("Using load balance mode", args.mode)

    ranks_vec = []
    mem_vec = []
    mem_max_vec = []
    mem_min_vec = []
    compute_lb_vec = []
    cell_creation_vec = []
    run_time_vec = []


    for n in range(6, 15, 2):
        print(f"Running with {n} MPI rank(s)...")
        ranks_vec.append(n)
        
        # Run the mpirun command
        cmd = [
            "mpirun", "-n", str(n), "special", "-mpi", "-python", "-s",
            "/Users/juanjose.garcia/dev/neurodamus/neurodamus/data/init.py",
            "--configFile=simulation_config.json", mode
        ]

        with open(f"log_{n}", "w") as logfile:
            subprocess.run(cmd, stdout=logfile, stderr=subprocess.STDOUT)

        with open(f"log_{n}", "r") as logfile:
            lines = logfile.readlines()

            mem = 0
            mem_max = 0
            mem_min = 0
            compute_lb = 0
            cell_creation = 0
            run_time = 0


            # Get Memusage (RSS)
            memusage_lines = [line for line in lines if "Memusage (RSS)" in line]
            if memusage_lines:
                matches = re.findall(r'=\s*([0-9.]+)', memusage_lines[-1])
                mem_max = float(matches[0])
                mem_min = float(matches[1])
                mem = float(matches[2])
                
            #get Compute LB
            match_lines = [line for line in lines if "Compute LB" in line]
            if match_lines:
                parts = match_lines[-1].split("|")
                if len(parts) > 2:
                    compute_lb = float(parts[2].strip())
            # Get Cell creation
            match_lines = [line for line in lines if "Cell creation" in line]
            if match_lines:
                parts = match_lines[-1].split("|")
                if len(parts) > 2:
                    cell_creation = float(parts[2].strip())
            #Get runtime
            match_lines = [line for line in lines if "finished Run" in line]
            if match_lines:
                parts = match_lines[-1].split("|")
                if len(parts) > 2:
                    run_time = float(parts[2].strip())


            if compute_lb == 0 and cell_creation == 0 and run_time == 0:
                msg = f"Simulation went wrong"
                print(f"\033[93mWARNING: something went wrong simulating with {n} ranks\033[0m")
        
            mem_vec.append(mem)
            mem_max_vec.append(mem_max)
            mem_min_vec.append(mem_min)
            compute_lb_vec.append(compute_lb)
            cell_creation_vec.append(cell_creation)
            run_time_vec.append(run_time)


            clean_files(["output", "simulation_config.json.SUCCESS"], ["pydamus_2025-"])
            if not args.log:
                clean_files([], ["log_"])
            if not args.cache:
                clean_files(["sim_conf", "mcomplex.dat", "cell_memory_usage.json"], ["allocation_r"])
        
    # print(f"Number of ranks: {ranks_vec}")
    # print(f"Memory usage mean: {mem_vec}")
    # print(f"Memory usage max: {mem_max_vec}")
    # print(f"Memory usage min: {mem_min_vec}")
    # print(f"Compute LB time: {compute_lb_vec}")
    # print(f"Cell creation time: {cell_creation_vec}")
    # print(f"Run time: {run_time_vec}")

    data_to_csv(f"result_{args.mode}.csv", ranks_vec, mem_vec, mem_max_vec, mem_min_vec, compute_lb_vec, cell_creation_vec, run_time_vec)
    print("Simulations finished!")