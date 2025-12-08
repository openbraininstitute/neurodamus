from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "src_dir": str(SIM_DIR / "usecase3"),
            "simconfig_file": "simulation_sonata.json",
            "extra_config": {
                "target_simulator": "NEURON",
                "inputs": {
                    "ex_efields": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "Mosaic",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_efields_stimulus_neuron(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus

    config_file = create_tmp_simulation_config_file
    nd = Neurodamus(config_file, disable_reports=True)
    nd.run()

    # compare spikes with refs
    ref_spike_gids = np.array(
        [
            0.0,
            1.0,
            2.0,
            0.0,
            1.0,
            2.0,
            0.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            0.0,
            2.0,
            0.0,
            2.0,
            1.0,
            0.0,
            2.0,
            0.0,
            1.0,
        ]
    )
    ref_timestamps = np.array(
        [
            0.2,
            0.3,
            0.3,
            2.5,
            3.4,
            4.2,
            7.2,
            7.3,
            8.5,
            13.8,
            19.5,
            25.6,
            28.8,
            32.0,
            35.7,
            38.5,
            40.3,
            41.9,
            45.1,
            47.7,
            49.5,
        ]
    )
    obtained_timestamps = nd._spike_vecs[0][0].as_numpy()
    obtained_spike_gids = nd._spike_vecs[0][1].as_numpy()
    npt.assert_allclose(obtained_spike_gids, ref_spike_gids)
    npt.assert_allclose(obtained_timestamps, ref_timestamps)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "src_dir": str(SIM_DIR / "usecase3"),
            "simconfig_file": "simulation_sonata.json",
            "extra_config": {
                "target_simulator": "CORENEURON",
                "inputs": {
                    "ex_efields": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "Mosaic",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_coreneuron_exception(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus

    config_file = create_tmp_simulation_config_file
    with pytest.raises(
        RuntimeError,
        match="CoreNEURON cannot simulate a model that contains the extracellular mechanism",
    ):
        Neurodamus(config_file, disable_reports=True)
