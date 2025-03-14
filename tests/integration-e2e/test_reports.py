import json
import pytest
from pathlib import Path


SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "v5_sonata"


def _read_sonata_report(report_file):
    import libsonata
    report = libsonata.ElementReportReader(report_file)
    pop_name = report.get_population_names()[0]
    node_ids = report[pop_name].get_node_ids()
    data = report[pop_name].get()
    return node_ids, data


def _create_reports_config(original_config_path: Path, tmp_path: Path) -> tuple[Path, Path]:
    """
    Create a modified configuration file in a temporary directory.
    """
    # Read the original config file
    with open(original_config_path, 'r') as f:
        config = json.load(f)

    # Update the network path in the config
    config["network"] = str(SIM_DIR / "sub_mini5" / "circuit_config.json")

    # Modify the necessary fields
    config["reports"] = config.get("reports", {})
    config["reports"]["summation_report"] = {
        "type": "summation",
        "cells": "Mosaic",
        "variable_name": "i_membrane,IClamp",
        "sections": "all",
        "dt": 0.1,
        "start_time": 0.0,
        "end_time": 40.0
    }
    config["reports"]["synapse_report"] = {
        "type": "synapse",
        "cells": "Mosaic",
        "variable_name": "ProbAMPANMDA_EMS.g",
        "sections": "all",
        "dt": 0.1,
        "start_time": 0.0,
        "end_time": 40.0
    }
    # Added to verify no exception is raised when point process is not present in a section
    config["reports"]["summation_ProbGABAAB"] = {
        "type": "summation",
        "cells": "Mosaic",
        "variable_name": "ProbGABAAB_EMS.i",
        "sections": "all",
        "dt": 0.1,
        "start_time": 0.0,
        "end_time": 40.0
    }

    # Write the modified configuration to a temporary file in tmp_path
    temp_config_path = tmp_path / "reports_config.json"
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=4)

    output_dir = Path(config["output"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = tmp_path / output_dir

    return str(temp_config_path), str(output_dir)


@pytest.mark.slow
def test_v5_sonata_reports(tmp_path):
    import numpy.testing as npt
    from neurodamus import Neurodamus

    config_file = SIM_DIR / "simulation_config_mini.json"
    temp_config_path, output_dir = _create_reports_config(config_file, tmp_path)

    nd = Neurodamus(temp_config_path)
    nd.run()

    report_refs = {
        "soma_report.h5":
            [(10, 3, -64.92565), (128, 1, -60.309418), (333, 4, -39.864296)],
        "summation_report.h5":
            [(20, 153, 1.19864846e-4), (60, 42, 1.1587787e-4), (283, 121, 3.3678625e-5)]
    }
    node_id_refs = [0, 1, 2, 3, 4]

    # Go through each report and compare the results
    for report_name, refs in report_refs.items():
        result_ids, result_data = _read_sonata_report(Path(output_dir) / report_name)
        assert result_ids == node_id_refs
        for row, col, ref in refs:
            npt.assert_allclose(result_data.data[row][col], ref)
