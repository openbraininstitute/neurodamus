import json
import os
from pathlib import Path
import pytest
import subprocess
import tempfile
import filecmp

from ..conftest import RINGTEST


tstop = 26

def check_outdat(REFERENCE_DIR):
    """ Compare outdat files to reference """
    OUTPUT_DIR = Path("output")
    filename = "out.dat"
    assert filecmp.cmp(OUTPUT_DIR / filename, REFERENCE_DIR / filename, shallow=False)

def check_checkpoint(REFERENCE_DIR):
    """ Compare save files to reference """
    OUTPUT_DIR = Path("output") / f"save_{REFERENCE_DIR.name}"
    filename = "1_2.dat"
    assert filecmp.cmp(OUTPUT_DIR /  filename, REFERENCE_DIR / filename, shallow=False)
    filename = "time.dat"
    assert filecmp.cmp(OUTPUT_DIR /  filename, REFERENCE_DIR / filename, shallow=False)

def check_cellstate(REFERENCE):
    """ Compare cellstates to reference """
    assert filecmp.cmp(REFERENCE.name, REFERENCE, shallow=False)


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
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
            },
            "run": {
                "tstop": tstop,
            },
        }
    }
], indirect=True)
def test_base_case(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus

    nd = Neurodamus(create_tmp_simulation_config_file, save="output/checkpoint", save_time=13)
    nd.run()
    # assert False
    

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#             "inputs": {
#                 "Stimulus": {
#                     "module": "pulse",
#                     "input_type": "current_clamp",
#                     "delay": 5,
#                     "duration": 50,
#                     "node_set": "RingA",
#                     "represents_physical_electrode": True,
#                     "amp_start": 10,
#                     "width": 1,
#                     "frequency": 50
#                 }
#             },
#             "run": {
#                 "tstop": tstop,
#             },
#         }
#     }
# ], indirect=True)
# def test_base_case(create_tmp_simulation_config_file):
#     nrn_t0 = f"{0:.6f}"
#     nrn_tstop = f"{tstop:.6f}"

#     # check dump state
#     for i in [0, 1, 2, 1000, 1001]:
#         command = ["neurodamus", "simulation_config.json", f"--dump-cell-state={i}"]
#         subprocess.run(command, check=True, capture_output=True)
#         # check cellstate i
#         check_cellstate(RINGTEST / "reference_save_restore" / nrn_t0 / f"{i+1}_cpu_t{nrn_t0}.corenrn")
#         check_cellstate(RINGTEST / "reference_save_restore" / nrn_tstop / f"{i+1}_cpu_t{nrn_tstop}.corenrn")

#     command = ["neurodamus", "simulation_config.json",f"--save=output/save_{nrn_tstop}"]
#     subprocess.run(command, check=True, capture_output=True)
#     # test checkpoint
#     check_outdat(RINGTEST / "reference_save_restore" / nrn_tstop)
#     check_checkpoint(RINGTEST / "reference_save_restore" / nrn_tstop)


# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#             "inputs": {
#                 "Stimulus": {
#                     "module": "pulse",
#                     "input_type": "current_clamp",
#                     "delay": 5,
#                     "duration": 50,
#                     "node_set": "RingA",
#                     "represents_physical_electrode": True,
#                     "amp_start": 10,
#                     "width": 1,
#                     "frequency": 50
#                 }
#             },
#             "run": {
#                 "tstop": tstop/2,
#             },
#         }
#     }
# ], indirect=True)
# def test_base_case_middle_point(create_tmp_simulation_config_file):
#     nrn_tstop = f"{tstop/2:.6f}"

#     # check dump state
#     for i in [0, 1, 2, 1000, 1001]:
#         command = ["neurodamus", "simulation_config.json", f"--dump-cell-state={i}"]
#         subprocess.run(command, check=True, capture_output=True)
#         # check cellstate i
#         check_cellstate(RINGTEST / "reference_save_restore" / nrn_tstop / f"{i+1}_cpu_t{nrn_tstop}.corenrn")

#     command = ["neurodamus", "simulation_config.json",f"--save=output/save_{nrn_tstop}"]
#     subprocess.run(command, check=True, capture_output=True)
#     # test checkpoint
#     check_outdat(RINGTEST / "reference_save_restore" / nrn_tstop)
#     check_checkpoint(RINGTEST / "reference_save_restore" / nrn_tstop)


# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#             "inputs": {
#                 "Stimulus": {
#                     "module": "pulse",
#                     "input_type": "current_clamp",
#                     "delay": 5,
#                     "duration": 50,
#                     "node_set": "RingA",
#                     "represents_physical_electrode": True,
#                     "amp_start": 10,
#                     "width": 1,
#                     "frequency": 50
#                 }
#             },
#             "run": {
#                 "tstop": tstop,
#             },
#         }
#     }
# ], indirect=True)
# def test_base_case_overshoot_middle_point(create_tmp_simulation_config_file):
#     nrn_tstop = f"{tstop:.6f}"
#     nrn_tsave = f"{tstop/2:.6f}"


#     command = ["neurodamus", "simulation_config.json",f"--save=output/save_{nrn_tsave}", f"--save-time={nrn_tsave}"]
#     subprocess.run(command, check=True, capture_output=True)
    # # outdat is emitted at the end
    # check_outdat(RINGTEST / "reference_save_restore" / nrn_tstop)
    # # test checkpoint
    # check_checkpoint(RINGTEST / "reference_save_restore" / nrn_tsave)


# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#             "inputs": {
#                 "Stimulus": {
#                     "module": "pulse",
#                     "input_type": "current_clamp",
#                     "delay": 5,
#                     "duration": 50,
#                     "node_set": "RingA",
#                     "represents_physical_electrode": True,
#                     "amp_start": 10,
#                     "width": 1,
#                     "frequency": 50
#                 }
#             },
#             "run": {
#                 "tstop": tstop,
#             },
#         }
#     }
# ], indirect=True)
# def test_save_time_full(create_tmp_simulation_config_file):
#     tsave = tstop

#     command = ["neurodamus", "simulation_config.json",f"--save=output/checkpoint", f"--save-time={tsave}"]
#     subprocess.run(command, check=True, capture_output=True)

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#             "inputs": {
#                 "Stimulus": {
#                     "module": "pulse",
#                     "input_type": "current_clamp",
#                     "delay": 5,
#                     "duration": 50,
#                     "node_set": "RingA",
#                     "represents_physical_electrode": True,
#                     "amp_start": 10,
#                     "width": 1,
#                     "frequency": 50
#                 }
#             },
#             "run": {
#                 "tstop": tstop,
#             },
#         }
#     }
# ], indirect=True)
# def test_save_time_overshoot(create_tmp_simulation_config_file):
#     tsave = tstop/2

#     command = ["neurodamus", "simulation_config.json",f"--save=output/checkpoint", f"--save-time={tsave}"]
#     subprocess.run(command, check=True, capture_output=True)

# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "simconfig_fixture": "ringtest_baseconfig",
#         "extra_config": {
#             "target_simulator": "CORENEURON",
#             "inputs": {
#                 "Stimulus": {
#                     "module": "pulse",
#                     "input_type": "current_clamp",
#                     "delay": 5,
#                     "duration": 50,
#                     "node_set": "RingA",
#                     "represents_physical_electrode": True,
#                     "amp_start": 10,
#                     "width": 1,
#                     "frequency": 50
#                 }
#             },
#             "run": {
#                 "tstop": tstop/2,
#             },
#         }
#     }
# ], indirect=True)
# def test_save_time_half(create_tmp_simulation_config_file):
#     tsave = tstop/2

#     command = ["neurodamus", "simulation_config.json",f"--save=output/checkpoint", f"--save-time={tsave}"]
#     subprocess.run(command, check=True, capture_output=True)