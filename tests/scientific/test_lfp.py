import pytest
import h5py
import numpy as np
from pathlib import Path
from ..conftest import RINGTEST_DIR

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"
LFP_3ELEC_RINGA_FILE = str(RINGTEST_DIR / "lfp_3elec_ringA.h5")
LFP_2ELEC_CELL0_FILE = str(RINGTEST_DIR / "lfp_2elec_ringA_cell0.h5")

_COMMON_INPUTS = {
    "stimulus_pulse": {
        "module": "pulse",
        "input_type": "current_clamp",
        "delay": 1,
        "duration": 50,
        "node_set": "RingA",
        "represents_physical_electrode": True,
        "amp_start": 10,
        "width": 1,
        "frequency": 50,
    }
}



@pytest.fixture
def test_weights_file(tmp_path):
    """Generate a synthetic LFP weights file for v5_sonata tests."""
    populations = {
        "default": [42, 0, 4],
        "other_pop": [77777, 88888]
    }

    test_file = h5py.File(tmp_path / "test_file.h5", 'w')

    for population, gids in populations.items():
        population_group = test_file.create_group(population)
        population_group.create_dataset("node_ids", data=gids)

        sec_ids_count = [2, 82, 140]
        total_segments = 224
        offsets = np.append(np.add.accumulate(sec_ids_count) - sec_ids_count, total_segments)
        population_group.create_dataset('offsets', data=offsets)

        electrodes_group = test_file.create_group("electrodes/" + population)
        matrix = []
        matrix.append([0.1, 0.2])
        matrix.append([0.3, 0.4])

        incrementx = 0.0
        incrementy = 0.0
        for i in range(2, total_segments):
            value_x = 0.4 + incrementx
            value_y = 0.3 + incrementy
            matrix.append([value_x, value_y])
            incrementx += 0.001
            incrementy -= 0.0032
        electrodes_group.create_dataset("scaling_factors", dtype='f8', data=matrix)

    return test_file, str(tmp_path / "test_file.h5")


def _read_sonata_lfp_file(lfp_file):
    import libsonata
    report = libsonata.ElementReportReader(lfp_file)
    lfp_data = {}
    for pop_name in report.get_population_names():
        node_ids = report[pop_name].get_node_ids()
        data = report[pop_name].get()
        lfp_data[pop_name] = (node_ids, data)
    return lfp_data


def test_v5_sonata_lfp(test_weights_file, create_simulation_config_file_factory, tmp_path):
    import numpy.testing as npt
    import json
    from neurodamus import Neurodamus
    from neurodamus.core.coreneuron_configuration import CoreConfig

    _, lfp_weights_file = test_weights_file
    with open(str(SIM_DIR / "v5_sonata" / "simulation_config_mini.json")) as f:
        sim_config_data = json.load(f)
    params = {
        "extra_config": {
            "network": str(SIM_DIR / "v5_sonata" / "sub_mini5" / "circuit_config.json"),
            "target_simulator": "CORENEURON",
            "reports": {
                "override_field": 1,
                "lfp": {
                    "type": "lfp",
                    "cells": "Mosaic",
                    "electrodes_file": lfp_weights_file,
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 1.0
                }
            }
        }
    }
    config_file = create_simulation_config_file_factory(params, tmp_path, sim_config_data)

    nd = Neurodamus(config_file)
    nd.run()

    # compare results with refs
    t3_data = np.array([0.00027065672, -0.00086610153, 0.0014563566, -0.0046603414])
    t7_data = np.array([0.00029265403, -0.0009364929, 0.001548515, -0.004955248])
    node_ids = np.array([0, 4])
    result_ids, result_data = _read_sonata_lfp_file(
        Path(CoreConfig.output_root) / "lfp.h5")["default"]

    npt.assert_allclose(result_data.data[3], t3_data)
    npt.assert_allclose(result_data.data[7], t7_data)
    npt.assert_allclose(result_ids, node_ids)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "reports": {
                "lfp_report": {
                    "type": "lfp",
                    "cells": "Mosaic",
                    "electrodes_file": str(RINGTEST_DIR / "lfp_file.h5"),
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 2.0
                }
            },
            "inputs": {
                "stimulus_pulse": {
                    "module": "pulse",
                    "input_type": "current_clamp",
                    "delay": 1,
                    "duration": 50,
                    "node_set": "RingA",
                    "represents_physical_electrode": True,
                    "amp_start": 10,
                    "width": 1,
                    "frequency": 50
                }
            }
        }
    },
], indirect=True)
def test_ringcircuit_lfp(create_tmp_simulation_config_file):
    import numpy.testing as npt
    from neurodamus import Neurodamus
    from neurodamus.core.coreneuron_configuration import CoreConfig

    nd = Neurodamus(create_tmp_simulation_config_file)
    nd.run()

    # compare results with refs
    lfp_data = _read_sonata_lfp_file(Path(CoreConfig.output_root) / "lfp_report.h5")
    result_ids, result_data = lfp_data["RingA"]

    node_ids = np.array([0, 1, 2])
    t11_data = np.array([0.11541528, 0.12541528, 0.6154153, 0.62541527, 1.1154152, 1.1254153])
    t19_data = np.array([0.11362588, 0.12362587, 0.6136259, 0.6236259, 1.1136259, 1.1236259])

    npt.assert_allclose(result_ids, node_ids)
    npt.assert_allclose(result_data.data[11], t11_data)
    npt.assert_allclose(result_data.data[19], t19_data)

    result_ids, result_data = lfp_data["RingB"]

    node_ids = np.array([0, 1])
    t11_data = np.array(
        [6.4121537e-07, 6.4121537e-07, 6.4121537e-07, 6.4121537e-07, 6.4121537e-07, 6.4121537e-07])
    t19_data = np.array(
        [8.2200177e-07, 8.2200177e-07, 8.2200177e-07, 8.2200177e-07, 8.2200177e-07, 8.2200177e-07])

    npt.assert_allclose(result_ids, node_ids)
    npt.assert_allclose(result_data.data[11], t11_data)
    npt.assert_allclose(result_data.data[19], t19_data)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "reports": {
                "lfp_report_A": {
                    "type": "lfp",
                    "cells": "RingA",
                    "electrodes_file": LFP_3ELEC_RINGA_FILE,
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 2.0,
                }
            },
            "inputs": _COMMON_INPUTS,
        }
    },
], indirect=True)
@pytest.mark.forked
def test_multi_lfp_report_single_A(create_tmp_simulation_config_file):
    """Run with only report A (RingA, 3 electrodes) and compare to reference."""
    import numpy.testing as npt
    from neurodamus import Neurodamus
    from neurodamus.core.coreneuron_configuration import CoreConfig

    nd = Neurodamus(create_tmp_simulation_config_file)
    nd.run()

    lfp_data = _read_sonata_lfp_file(Path(CoreConfig.output_root) / "lfp_report_A.h5")
    result_ids, result_data = lfp_data["RingA"]

    assert list(result_ids) == [0, 1, 2]
    assert result_data.data.shape[1] == 9  # 3 gids * 3 electrodes

    ref = _read_sonata_lfp_file(
        str(RINGTEST_DIR / "reference" / "lfp_reports" / "lfp_single_A.h5"))["RingA"]
    npt.assert_allclose(result_data.data, ref[1].data, rtol=1e-5)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "reports": {
                "lfp_report_B": {
                    "type": "lfp",
                    "cells": "RingA_Cell0",
                    "electrodes_file": LFP_2ELEC_CELL0_FILE,
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 2.0,
                }
            },
            "inputs": _COMMON_INPUTS,
        }
    },
], indirect=True)
@pytest.mark.forked
def test_multi_lfp_report_single_B(create_tmp_simulation_config_file):
    """Run with only report B (RingA_Cell0, 2 electrodes) and compare to reference."""
    import numpy.testing as npt
    from neurodamus import Neurodamus
    from neurodamus.core.coreneuron_configuration import CoreConfig

    nd = Neurodamus(create_tmp_simulation_config_file)
    nd.run()

    lfp_data = _read_sonata_lfp_file(Path(CoreConfig.output_root) / "lfp_report_B.h5")
    result_ids, result_data = lfp_data["RingA"]

    assert list(result_ids) == [0]
    assert result_data.data.shape[1] == 2  # 1 gid * 2 electrodes

    ref = _read_sonata_lfp_file(
        str(RINGTEST_DIR / "reference" / "lfp_reports" / "lfp_single_B.h5"))["RingA"]
    npt.assert_allclose(result_data.data, ref[1].data, rtol=1e-5)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "reports": {
                "lfp_report_A": {
                    "type": "lfp",
                    "cells": "RingA",
                    "electrodes_file": LFP_3ELEC_RINGA_FILE,
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 2.0,
                },
                "lfp_report_B": {
                    "type": "lfp",
                    "cells": "RingA_Cell0",
                    "electrodes_file": LFP_2ELEC_CELL0_FILE,
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 2.0,
                },
            },
            "inputs": _COMMON_INPUTS,
        }
    },
], indirect=True)
@pytest.mark.forked
def test_multi_lfp_report_combined(create_tmp_simulation_config_file):
    """Run with both reports and verify each matches its single-report run."""
    import numpy.testing as npt
    from neurodamus import Neurodamus
    from neurodamus.core.coreneuron_configuration import CoreConfig

    nd = Neurodamus(create_tmp_simulation_config_file)
    nd.run()

    # Report A: RingA gids 0,1,2 with 3 electrodes
    lfp_A = _read_sonata_lfp_file(Path(CoreConfig.output_root) / "lfp_report_A.h5")
    result_ids_A, result_data_A = lfp_A["RingA"]
    assert list(result_ids_A) == [0, 1, 2]
    assert result_data_A.data.shape[1] == 9  # 3 gids * 3 electrodes

    # Report B: RingA_Cell0 gid 0 with 2 electrodes
    lfp_B = _read_sonata_lfp_file(Path(CoreConfig.output_root) / "lfp_report_B.h5")
    result_ids_B, result_data_B = lfp_B["RingA"]
    assert list(result_ids_B) == [0]
    assert result_data_B.data.shape[1] == 2  # 1 gid * 2 electrodes

    # Compare against single-report references
    ref_A = _read_sonata_lfp_file(
        str(RINGTEST_DIR / "reference" / "lfp_reports" / "lfp_single_A.h5"))["RingA"]
    ref_B = _read_sonata_lfp_file(
        str(RINGTEST_DIR / "reference" / "lfp_reports" / "lfp_single_B.h5"))["RingA"]
    npt.assert_allclose(result_data_A.data, ref_A[1].data, rtol=1e-5,
                        err_msg="Report A differs from single-report reference")
    npt.assert_allclose(result_data_B.data, ref_B[1].data, rtol=1e-5,
                        err_msg="Report B differs from single-report reference")


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "v5_sonata_config",
        "extra_config": {
            "target_simulator": "CORENEURON",
            "reports": {
                "override_field": 1,
                "soma_report": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "v",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 1.0
                }
            }
        }
    },
], indirect=True)
@pytest.mark.forked
def test_v5_coreneuron_no_lfp_smoke(create_tmp_simulation_config_file):
    """Smoke test: switch to parametrize fixture."""
    from neurodamus import Neurodamus
    nd = Neurodamus(create_tmp_simulation_config_file)
    nd.run()
