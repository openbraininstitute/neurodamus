import pytest
import numpy
import os




@pytest.fixture
def injection_config_file(ringtest_baseconfig):
    from tempfile import NamedTemporaryFile
    import json
  
    ringtest_baseconfig["inputs"] = {
        "Stimulus": {
            "module": "noise",
            "input_type": "current_clamp",
            "delay": 5,
            "duration": 100,
            "node_set": "RingA",
            "represents_physical_electrode": False,
            "mean": 200,
            "variance": 0.1
        }
    }

    # ringtest_baseconfig["inputs"] = {
    #     "Stimulus": {
    #         "module": "pulse",
    #         "input_type": "current_clamp",
    #         "delay": 1,
    #         "duration": 100,
    #         "node_set": "RingA",
    #         "represents_physical_electrode": True,
    #         "amp_start": 1000,
    #         "width": 10,
    #         "frequency": 50, 
    #     }
    # }

    with NamedTemporaryFile("w", suffix='.json', delete=False) as config_file:
        json.dump(ringtest_baseconfig, config_file)
    yield config_file
    os.unlink(config_file.name)



# @pytest.mark.forked
def test_current_injection(injection_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd

    nd = Neurodamus(injection_config_file.name,)
   

    cell_id_ringA = 1
    manager_ringA = nd.circuits.get_node_manager("RingA")
    cell_ringA = manager_ringA.get_cell(cell_id_ringA)
    voltage_ringA = Nd.Vector()
    voltage_ringA.record(cell_ringA._cellref.soma[0](0.5)._ref_v, 0.125)
    

    cell_id_ringB = 1001
    manager_ringB = nd.circuits.get_node_manager("RingB")
    cell_ringB = manager_ringB.get_cell(cell_id_ringB)
    voltage_ringB = Nd.Vector()
    voltage_ringB.record(cell_ringB._cellref.soma[0](0.5)._ref_v)


    Nd.finitialize()  # reinit for the recordings to be registered
    nd.run()  # until end
    voltage_ringA = voltage_ringA.as_numpy()
    voltage_ringB = voltage_ringB.as_numpy()

    print("-------------------------------")
    print("-------------------------------")
    print("-------------------------------")
    ringA_timestamps = nd._spike_vecs[0][0].as_numpy()
    ringA_spike_gids = nd._spike_vecs[0][1].as_numpy().astype(int)
    print("Spikes ringA", ringA_spike_gids, ringA_timestamps)

    # print("Voltage ringA:", voltage_ringA)
    # print("Maximum voltage ringA cell 1:   ", numpy.max(voltage_ringA))
    # print("Minimum voltage ringA cell 1:   ", numpy.min(voltage_ringA))

    
    # print("Voltage ringB:", voltage_ringB)
    print("Maximum voltage ringB cell 1001:", numpy.max(voltage_ringB))
    print("Minimum voltage ringB cell 1001:", numpy.min(voltage_ringB))


    # print(nd._spike_populations)





