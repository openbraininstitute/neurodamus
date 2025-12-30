import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import libsonata
import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def sonata_config_files(sonata_config, input_type, tmp_path):
    def create_config(represents_physical_electrode):
        # Create a deep copy of sonata_config for each configuration to avoid conflicts
        config_copy = json.loads(json.dumps(sonata_config))

        stimulus_config = {
            "input_type": input_type,
            "delay": 5,
            "duration": 2100,
            "node_set": "l4pc",
            "represents_physical_electrode": represents_physical_electrode
        }

        if input_type == "current_clamp":
            stimulus_config.update({
                "module": "noise",
                "mean": 0.05,
                "variance": 0.01
            })
        elif input_type == "conductance":
            stimulus_config.update({
                "module": "ornstein_uhlenbeck",
                "mean": 0.05,
                "sigma": 0.01,
                "tau": 0.1
            })

        config_copy["inputs"] = {"Stimulus": stimulus_config}
        config_copy["reports"] = {
            "current": {
                "type": "summation",
                "cells": "l4pc",
                "variable_name": "i_membrane",
                "unit": "nA",
                "dt": 0.1,
                "start_time": 0.0,
                "end_time": 50.0
            },
            "voltage": {
                "type": "compartment",
                "cells": "l4pc",
                "variable_name": "v",
                "unit": "mV",
                "dt": 0.1,
                "start_time": 0.0,
                "end_time": 50.0
            }
        }
        return config_copy

    true_path = (tmp_path / "true_config.json")
    with true_path.open("w") as fd:
        json.dump(create_config(represents_physical_electrode=True), fd)

    false_path = (tmp_path / "false_config.json")
    with false_path.open("w") as fd:
        json.dump(create_config(represents_physical_electrode=False), fd)

    return false_path, true_path


def _read_sonata_soma_report(report_name):
    report = libsonata.SomaReportReader(report_name)
    pop_name = report.get_population_names()[0]
    ids = report[pop_name].get_node_ids()
    data = report[pop_name].get(node_ids=[ids[0]])
    return np.array(data.data).flatten()


def _run_simulation(config_file):
    output_dir = "output_current_conductance"
    command = [
        "neurodamus",
        config_file,
        f"--output-path={output_dir}"
    ]
    config_dir = Path(config_file).parent
    subprocess.run(command, cwd=config_dir, check=True)
    soma_report_path = config_dir / output_dir / "voltage.h5"
    return _read_sonata_soma_report(soma_report_path)


@pytest.mark.parametrize("input_type", [
    "current_clamp",
    "conductance",
])
def test_current_conductance_injection(sonata_config_files):
    """
    Test the consistency of voltage traces between original and new configurations
    (set by 'represents_physical_electrode': true/false)
    under different types of input (current clamp and conductance).
    """
    config_file_original, config_file_new = sonata_config_files

    voltage_vec_original = _run_simulation(str(config_file_original))
    voltage_vec_new = _run_simulation(str(config_file_new))

    npt.assert_equal(voltage_vec_original, voltage_vec_new)
