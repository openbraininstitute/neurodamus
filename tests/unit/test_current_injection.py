import pytest
import numpy
import os




@pytest.fixture
def injection_config_file(ringtest_baseconfig):
    from tempfile import NamedTemporaryFile
    import json
  
    # ringtest_baseconfig["inputs"] = {
    #     "Stimulus": {
    #         "module": "noise",
    #         "input_type": "current_clamp",
    #         "delay": 5,
    #         "duration": 100,
    #         "node_set": "RingA",
    #         "represents_physical_electrode": True,
    #         "mean_percent": 75,
    #         "variance": 0.1
    #     }
    # }

    ringtest_baseconfig["inputs"] = {
        "Stimulus": {
            "module": "pulse",
            "input_type": "current_clamp",
            "delay": 5,
            "duration": 100,
            "node_set": "RingA",
            "represents_physical_electrode": True,
            "amp_start": 1000,
            "width": 1,
            "frequency": 100, 
        }
    }

    with NamedTemporaryFile("w", suffix='.json', delete=False) as config_file:
        json.dump(ringtest_baseconfig, config_file)
    yield config_file
    os.unlink(config_file.name)



@pytest.mark.forked
def test_current_injection(injection_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core.configuration import SimConfig

    nd = Neurodamus(
        injection_config_file.name,
        # output_path="/Users/juanjose.garcia/dev/neurodamus/output",
        # coreneuron_direct_mode=True
        # disable_reports=True,
        # cleanup_atexit=False,
    )
    nd.run()  # until end


    print(nd._spike_populations)

    ringA_timestamps = nd._spike_vecs[0][0].as_numpy()
    ringA_spike_gids = nd._spike_vecs[0][1].as_numpy().astype(int)

    ringB_timestamps = nd._spike_vecs[1][0].as_numpy()
    ringB_spike_gids = nd._spike_vecs[1][1].as_numpy().astype(int)

    assert(len(ringA_timestamps) == len(ringA_spike_gids))
    assert(len(ringB_timestamps) == len(ringB_spike_gids))

    print("-------------------------------")
    print(ringA_spike_gids, ringA_timestamps)
    print(ringB_spike_gids, ringB_timestamps)



