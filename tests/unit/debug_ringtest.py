import os
import json


from tests import utils
from tests import conftest
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



@utils.change_directory('debug_ringtest')
def debug_test():
    # Define dummy data for the JSON file
    simulation_config = {
        "network": "../tests/simulations/ringtest/circuit_config.json",
        "node_sets_file": "../tests/simulations/ringtest/nodesets.json",
        "target_simulator": "NEURON",
        "run": {
            "random_seed": 1124,
            "dt": 0.1,
            "tstop": 600
        },
        "node_set": "Mosaic",
        "conditions": {
            "celsius": 35,
            "v_init": -65
        },
        "inputs": {
                "Stimulus": {
                    "module": "linear",
                    "input_type": "current_clamp",
                    "delay": 5,
                    "duration": 1,
                    "node_set": "RingA",
                    "represents_physical_electrode": True,
                    "amp_start": 5
                }
            },
        "connection_overrides": [
                {
                    "name": "A2A",
                    "source": "RingA",
                    "target": "RingA",
                    "modoverride": "DetAMPANMDA",
                    "synapse_delay_override": 15,
                },
                {
                    "name": "B2B",
                    "source": "RingB",
                    "target": "RingB",
                    "modoverride": "DetAMPANMDA",
                    "synapse_delay_override": 15,
                },
                {
                    "name": "A2B",
                    "source": "RingA",
                    "target": "RingB",
                    "modoverride": "DetAMPANMDA",
                    "synapse_delay_override": 15,
                },
            ]
        }
    
    # Write the dummy data to a JSON file inside the base folder
    json_file_path = 'simulation_config.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(simulation_config, json_file, indent=4)

    #### start of the test
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Ndc

    nd = Neurodamus("simulation_config.json")

    def record_traces(name, pop, cell, sec, d, q):
        key = (pop, cell.gid, sec)
        if key in d[name]:
            return
        d[name][key] = Ndc.Vector().record(q)


    connections = [
        ("RingA", 1, "RingA", 2),
        ("RingA", 2, "RingA", 3),
        ("RingA", 3, "RingA", 1),
        ("RingB", 1, "RingB", 2),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    traces = {}
    traces["voltage"] = {}
    traces["nc_spike"] = {}


    for src_pop, src_raw_gid, tgt_pop, tgt_raw_gid in connections:
        src_gid, tgt_gid, edges, selection = utils.get_edge_data(
            nd, src_pop, src_raw_gid, tgt_pop, tgt_raw_gid)
        src_cell, tgt_cell = nd._pc.gid2cell(src_gid), nd._pc.gid2cell(tgt_gid)

        for sec in ["soma", "dend"]:
            record_traces("voltage", src_pop, src_cell, sec, traces, getattr(src_cell, sec)[0](0.5)._ref_v)
            record_traces("voltage", tgt_pop, tgt_cell, sec, traces, getattr(tgt_cell, sec)[0](0.5)._ref_v)

        nclist = Ndc.cvode.netconlist(src_cell, tgt_cell, "")
        if len(nclist):
            nc = nclist[0]
            traces["nc_spike"][(src_pop, src_cell.gid, tgt_pop, tgt_cell.gid)] = Ndc.Vector()
            nc.record(traces["nc_spike"][(src_pop, src_cell.gid, tgt_pop, tgt_cell.gid)])
            utils.inspect(nc)

    traces['t'] = Ndc.Vector().record(Ndc._ref_t)
    Ndc.finitialize()
    nd.run(False) # malloc fails if this is true. nd is too aggressive in removing stuff?

    for k, v in traces["nc_spike"].items():
        print(k, list(v))

    return traces


def postproc(traces):
    time = np.array(traces['t'])
    # Extract unique (pop, id) pairs from the voltage trace keys
    unique_pairs = sorted(set((key[0], key[1]) for key in traces["voltage"]))

    num_subplots = len(unique_pairs)
    num_rows, num_cols = 2, 3  # Adjust grid as needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True)
    axes = axes.flatten()  # Flatten for iteration

    for ax, (pop, gid) in zip(axes, unique_pairs):
        # Plot voltage traces for each section
        for sec in ["soma", "dend"]:
            key = (pop, gid, sec)
            if key in traces["voltage"]:
                ax.plot(time, np.array(traces["voltage"][key]), label=sec)

        # Gather all netcons where this cell is the source
        src_connections = [
            ((src_pop, src_gid, tgt_pop, tgt_gid), spike_vec)
            for (src_pop, src_gid, tgt_pop, tgt_gid), spike_vec in traces["nc_spike"].items()
            if src_pop == pop and src_gid == gid
        ]

        idx = 0
        for (src_pop, src_gid, tgt_pop, tgt_gid), spike_vec in traces["nc_spike"].items():
            if (src_pop == pop and src_gid == gid):
                y_val = -30
                spike_times = np.array(list(spike_vec))
                ax.scatter(
                    spike_times,
                    np.full_like(spike_times, y_val),
                    marker='v',
                    color='blue',
                    label=f"{src_pop}:{src_gid} -> {tgt_pop}:{tgt_gid}"
                ) 
            if (tgt_pop == pop and tgt_gid == gid):
                y_val = idx
                idx = idx+10
                spike_times = np.array(list(spike_vec))
                ax.scatter(
                    spike_times,
                    np.full_like(spike_times, y_val),
                    label=f"{src_pop}:{src_gid} -> {tgt_pop}:{tgt_gid}"
                ) 

        # for idx, ((src_pop, src_gid, tgt_pop, tgt_gid), spike_vec) in enumerate(src_connections):
        #     spike_times = np.array(list(spike_vec))
        #     # if spike_times.size > 0:
        #     y_val = idx*10+5
        #     ax.scatter(
        #         spike_times,
        #         np.full_like(spike_times, y_val),
        #         marker='v',
        #         label=f"Spike -> {tgt_pop}:{tgt_gid}"
        #     )
            # else:
            #     print("no spikes! ", (src_pop, src_gid, tgt_pop, tgt_gid))

        ax.axhline(-30, linestyle="--", color="gray")  # Reference line
        ax.set_ylim(-80, 80)
        ax.set_ylabel('Membrane Potential (mV)')
        ax.set_title(f'{pop}, {gid}')
        ax.legend()
        ax.grid(True)

    # Hide unused subplots
    for ax in axes[len(unique_pairs):]:
        ax.axis('off')

    # Set x-labels on the bottom subplots
    axes[-2].set_xlabel('Time (ms)')
    axes[-1].set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.show()



# def postproc(traces):
#     time = np.array(traces['t'])

#     # Extract unique (pop, id) pairs from the traces keys
#     unique_pairs = sorted(set((key[0], key[1]) for key in traces["voltage"]))

#     num_subplots = len(unique_pairs)
#     num_rows, num_cols = 2, 3  # 2 rows, 3 columns
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True)

#     axes = axes.flatten()  # Flatten for easy iteration

#     for ax, (pop, gid) in zip(axes, unique_pairs):
#         for sec in ["soma", "dend"]:
#             key = (pop, gid, sec)
#             if key in traces["voltage"]:
#                 ax.plot(time, np.array(traces["voltage"][key]), label=sec)

#         ax.axhline(-30, linestyle="--", color="gray")  # Add horizontal dashed line
#         ax.set_ylim(-80, 80)  # Set y-axis limits
#         ax.set_ylabel('Membrane Potential (mV)')
#         ax.set_title(f'{pop}, {gid}')
#         ax.legend()
#         ax.grid(True)

#     # Hide any unused subplots
#     for ax in axes[len(unique_pairs):]:
#         ax.axis('off')

#     axes[-2].set_xlabel('Time (ms)')  # Set x-label on second-to-last row
#     axes[-1].set_xlabel('Time (ms)')  # Set x-label on last row

#     plt.tight_layout()
#     plt.show()




if __name__ == '__main__':
    traces = debug_test()
    postproc(traces)
    



