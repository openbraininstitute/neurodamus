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
            "random_seed": 1122,
            "dt": 0.1,
            "tstop": 50
        },
        "node_set": "Mosaic",
        "conditions": {
            "celsius": 35,
            "v_init": -65
        },
        "inputs": {
                "Stimulus": {
                    "module": "pulse",
                    "input_type": "current_clamp",
                    "delay": 5,
                    "duration": 50,
                    "node_set": "RingA",
                    "represents_physical_electrode": True,
                    "amp_start": 10,
                    "width": 1,
                    "frequency": 50
                }
            }
        }
    
    # Write the dummy data to a JSON file inside the base folder
    json_file_path = 'simulation_config.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(simulation_config, json_file, indent=4)

    #### start of the test
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Ndc

    nd = Neurodamus("simulation_config.json")

    def record(pop, cell, sec, d):
        key = (pop, cell.gid, sec)
        if key in d:
            return
        d[key] = Ndc.Vector().record(getattr(cell, sec)[0](0.5)._ref_v)


    connections = [
        ("RingA", 3, "RingA", 1),
        ("RingB", 2, "RingB", 1),
        ("RingA", 1, "RingB", 1),
    ]

    traces = {}


    for src_pop, src_raw_gid, tgt_pop, tgt_raw_gid in connections:
        src_gid, tgt_gid, edges, selection = utils.get_edge_data(
            nd, src_pop, src_raw_gid, tgt_pop, tgt_raw_gid)
        src_cell, tgt_cell = nd._pc.gid2cell(src_gid), nd._pc.gid2cell(tgt_gid)

        for sec in ["soma", "dend"]:
            record(src_pop, src_cell, sec, traces)
            record(tgt_pop, tgt_cell, sec, traces)

        # nclist = Ndc.cvode.netconlist(src_cell, tgt_cell, "")
        # for nc in nclist:
        #     utils.inspect(nc.weight[0])

    traces['t'] = Ndc.Vector().record(Ndc._ref_t)
    Ndc.finitialize()
    nd.run(False) # malloc fails if this is true. nd is too aggressive in removing stuff?

    return traces

def postproc(traces):
    # Extract the time and voltage traces
    time = np.array(traces['t'])
    
    # Create a figure with 2 subplots (one for soma and one for dendrite) with a smaller size
    fig, (ax_soma, ax_dend) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Iterate through the traces dictionary and plot each trace
    for key, trace in traces.items():
        if key != 't':  # Skip the time vector
            voltage = np.array(trace)
            if "soma" in key:
                ax_soma.plot(time, voltage, label=f'{key}')
            elif "dend" in key:
                ax_dend.plot(time, voltage, label=f'{key}')
    
    # Customize the soma subplot
    ax_soma.set_xlabel('')
    ax_soma.set_ylabel('Membrane Potential (mV)')
    ax_soma.set_title('Soma Voltage Traces')
    ax_soma.legend(loc='best')
    ax_soma.grid(True)

    # Customize the dendrite subplot
    ax_dend.set_xlabel('Time (ms)')
    ax_dend.set_ylabel('Membrane Potential (mV)')
    ax_dend.set_title('Dendrite Voltage Traces')
    ax_dend.legend(loc='best')
    ax_dend.grid(True)

    # Adjust layout to fit the screen better
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    traces = debug_test()
    postproc(traces)
    



