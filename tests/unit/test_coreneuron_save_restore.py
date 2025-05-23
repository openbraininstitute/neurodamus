"""Test the save and restore functionality of the simulation.

Note: additional creation of reports cannot be tested in a unit test as it requires
libsonatareport (which is missing in the unit test environment).
"""

import filecmp
import json
import subprocess
from pathlib import Path

import pytest

from tests import utils

checkpoint_content = {"time.dat", "1_2.dat", "populations_offset.dat"}
removable_checkpoint_content = {"report.conf", "sim.conf", "coreneuron_input"}
output_content = {"out.dat", "populations_offset.dat"}
coreneuron_input_content = {
    "1_1.dat",
    "1_2.dat",
    "1_3.dat",
    "bbcore_mech.dat",
    "files.dat",
    "globals.dat"}


def update_sim_conf(tstop, output_dir):
    """
    Update the simulation_config.json file with the new tstop and output_dir values.
    """
    with open("simulation_config.json", "r") as f:
        sim_config = json.load(f)
        sim_config["run"]["tstop"] = tstop
        sim_config["output"]["output_dir"] = output_dir
    with open("simulation_config.json", "w") as f:
        json.dump(sim_config, f, indent=2)


def check_dir_content(dir, files):
    """
    Check that the files (and only these files) are present
    in the directory.
    """
    dir = Path(dir)
    if not dir.exists():
        raise FileNotFoundError(f"Directory {dir} does not exist.")
    if not dir.is_dir():
        raise FileNotFoundError(f"{dir} is not a directory.")
    for f in files:
        if not (dir / f).exists():
            raise FileNotFoundError(f"File {f} does not exist in {dir}")
    for f in dir.iterdir():
        if f.name not in files:
            raise FileNotFoundError(f"File {f.name} exists in {dir} but is not expected.")


def check_report_conf(checkpoint_dir, substitutions):
    """
    Check that the report.conf file in the checkpoint directory
    has the expected end times.
    """
    found_keys = set()
    file_path = Path(checkpoint_dir) / "report.conf"
    with open(file_path, "rb") as f:
        lines = f.readlines()
        for line in lines:
            try:

                parts = line.decode().split()
                key = tuple(parts[0:2])  # Report name and target name

                if key in substitutions:
                    assert float(parts[9]) == pytest.approx(float(substitutions[key])), (
                        f"End time for {key} is not {substitutions[key]} in report.conf. "
                        f"It is {parts[9]} instead."
                    )
                    found_keys.add(key)
            except (UnicodeDecodeError, IndexError):
                # Ignore lines that cannot be decoded (binary data)
                continue
    # Check that all expected keys were found
    if len(substitutions) != len(found_keys):
        missing_keys = set(substitutions.keys()) - found_keys
        raise KeyError(f"The following keys were not found in {file_path}: {missing_keys}")


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
        }
    }
], indirect=True)
def test_file_placement_base(create_tmp_simulation_config_file):

    command = ["neurodamus", "simulation_config.json"]
    subprocess.run(command, check=True, capture_output=True)

    # Check the output directory: output + save files
    check_dir_content("output", output_content)
    assert not Path("build").exists()


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
        }
    }
], indirect=True)
def test_file_placement_keep_build(create_tmp_simulation_config_file):
    command = ["neurodamus", "simulation_config.json", "--keep-build"]
    subprocess.run(command, check=True, capture_output=True)

    # Check the output directory: output + save files
    check_dir_content("output", output_content)
    check_dir_content("build", removable_checkpoint_content)
    with pytest.raises(FileNotFoundError):
        check_dir_content("output/coreneuron_input", coreneuron_input_content)


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
        }
    }
], indirect=True)
def test_file_placement_save(create_tmp_simulation_config_file):

    command = ["neurodamus", "simulation_config.json", "--save=checkpoint"]
    subprocess.run(command, check=True, capture_output=True)

    # Check the output directory: output + save files
    check_dir_content("output", output_content)
    check_dir_content("checkpoint", checkpoint_content | removable_checkpoint_content)
    check_dir_content("checkpoint/coreneuron_input", coreneuron_input_content)


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
        }
    }
], indirect=True)
def test_file_placement_keep_build_save(create_tmp_simulation_config_file):

    command = ["neurodamus", "simulation_config.json", "--keep-build", "--save=checkpoint"]
    subprocess.run(command, check=True, capture_output=True)

    # Check the output directory: output + save files
    check_dir_content("output", output_content)
    check_dir_content("checkpoint", checkpoint_content | removable_checkpoint_content)
    check_dir_content("checkpoint/coreneuron_input", coreneuron_input_content)
    assert not Path("build").exists()


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
            "reports": {
                "soma_v": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "v",
                    "sections": "soma",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 18.0,
                },
                "compartment_i": {
                    "type": "compartment",
                    "cells": "Mosaic",
                    "variable_name": "i_membrane",
                    "sections": "all",
                    "dt": 1,
                    "start_time": 0.0,
                    "end_time": 40.0,
                },
            },
            "run": {
                "tstop": 0,
            },
            "output": {
                "output_dir": "output_0_0",
            }
        }
    }
], indirect=True)
def test_full_run_vs_save_restore(create_tmp_simulation_config_file):
    """
    Test the save and restore functionality of the simulation.

    This test performs the following steps:
    1. Runs a full simulation and dumps the cell states for all the GIDs.
    2. Updates the simulation configuration to run a partial simulation, saving a checkpoint.
    3. Restores the simulation from the checkpoint and dumps the cell states again.
    4. Compares the dumped cell states from the full run with those from the save-restore process.
    5. Compares the output data files (out.dat) from the full run and the save-restore process
       for consistency within specified time ranges.

    Note: this also implicitly tests that the save/restore works even if the save folder is
    not a subfolder of the output folder.
    """
    gids = [0, 1, 2, 1000, 1001]
    t = [0, 13, 26]

    update_sim_conf(t[2], f"output_{t[0]}_{t[2]}")

    for i in gids:
        command = ["neurodamus", "simulation_config.json", f"--dump-cell-state={i}"]
        subprocess.run(command, check=True, capture_output=True)

    update_sim_conf(t[1], f"output_{t[0]}_{t[1]}")

    command = ["neurodamus", "simulation_config.json", f"--save=checkpoint_{t[1]}"]
    subprocess.run(command, check=True, capture_output=True)

    # check result.conf end times
    report_times = {("soma_v.h5", "Mosaic"): t[1], ("compartment_i.h5", "Mosaic"): t[1]}
    check_report_conf(f"checkpoint_{t[1]}", report_times)

    update_sim_conf(t[2], f"output_{t[1]}_{t[2]}")

    for i in gids:
        command = [
            "neurodamus",
            "simulation_config.json",
            f"--dump-cell-state={i}",
            f"--restore=checkpoint_{t[1]}"]
        subprocess.run(command, check=True, capture_output=True)

    command = [
        "neurodamus",
        "simulation_config.json",
        f"--save=checkpoint_{t[2]}",
        f"--restore=checkpoint_{t[1]}"]
    subprocess.run(command, check=True, capture_output=True, text=True)

    # check result.conf end times
    report_times = {("soma_v.h5", "Mosaic"): 18, ("compartment_i.h5", "Mosaic"): t[2]}
    check_report_conf(f"checkpoint_{t[2]}", report_times)

    # compare celldump states
    full_run_dir = Path(f"output_{t[0]}_{t[2]}")
    save_restore_dir2 = Path(f"output_{t[1]}_{t[2]}")
    # Compare the files of the form 1_cpu_t<t>.corenrn
    for i in gids:
        file_name = f"{i+1}_cpu_t{t[2]:.6f}.corenrn"
        file1 = full_run_dir / file_name
        file2 = save_restore_dir2 / file_name
        if not file1.exists() or not file2.exists():
            raise FileNotFoundError(f"One or both files do not exist: {file1}, {file2}")
        # Compare the files
        assert filecmp.cmp(file1, file2, shallow=False)

    # Compare the out.dat files
    out_dat_file = "out.dat"
    full_run_out_dat = full_run_dir / out_dat_file
    save_restore_dir1 = Path(f"output_{t[0]}_{t[1]}")
    save_restore_out_dat1 = save_restore_dir1 / out_dat_file
    save_restore_out_dat2 = save_restore_dir2 / out_dat_file
    assert utils.compare_outdat_files(full_run_out_dat, save_restore_out_dat1, end_time=t[1],)
    assert utils.compare_outdat_files(full_run_out_dat, save_restore_out_dat2, start_time=t[1],)
