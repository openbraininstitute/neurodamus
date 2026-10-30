import json
import os
import subprocess
from pathlib import Path

import libsonata
import pytest

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "v5_sonata"
CONFIG_FILE_MINI = "simulation_config_mini.json"
CIRCUIT_DIR = "sub_mini5"


def _run_neurodamus(tmp_path, *args):
    """Run neurodamus in `tmp_path` and fail with captured output on non-zero exit."""
    result = subprocess.run(
        ["neurodamus", CONFIG_FILE_MINI, *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"neurodamus {' '.join(args)} exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _report_path(tmp_path, sc, name):
    path = Path(sc.report(name).file_name)
    return path if path.is_absolute() else tmp_path / path


def test_cli_disable_reports(tmp_path):
    with open(SIM_DIR / CONFIG_FILE_MINI, encoding="utf-8") as fd:
        sim_config_data = json.load(fd)

    sim_config_data["network"] = str(SIM_DIR / CIRCUIT_DIR / "circuit_config.json")
    config_path = tmp_path / CONFIG_FILE_MINI
    with open(config_path, "w", encoding="utf-8") as fd:
        json.dump(sim_config_data, fd)

    _run_neurodamus(tmp_path, "--disable-reports")

    sc = libsonata.SimulationConfig.from_file(str(config_path))
    spikes_path = tmp_path / sc.output.output_dir / "out.h5"

    # Spikes are present even if we disable reports
    assert spikes_path.is_file()

    for name in sc.list_report_names:
        report_path = _report_path(tmp_path, sc, name)
        assert not report_path.is_file(), f"File '{report_path}' should NOT exist."

    _run_neurodamus(tmp_path)
    assert spikes_path.is_file()
    for name in sc.list_report_names:
        report_path = _report_path(tmp_path, sc, name)
        assert report_path.is_file(), f"File '{report_path}' not found."


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "target_simulator": "CORENEURON",
        "extra_config": {
            "reports": {
                "soma_v": {
                    "type": "compartment",
                    "variable_name": "v",
                    "sections": "soma",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 40.0
                }
           },
        }
    },
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "target_simulator": "NEURON",
        "extra_config": {
            "reports": {
                "soma_v": {
                    "type": "compartment",
                    "variable_name": "v",
                    "sections": "soma",
                    "dt": 0.1,
                    "start_time": 0.0,
                    "end_time": 40.0
                }
           },
        }
    }
], indirect=True,
)
def test_cli_report_buff_size(create_tmp_simulation_config_file):
    custom_env = os.environ.copy()
    custom_env["SPDLOG_LEVEL"] = "debug"
    result = subprocess.run(
        ["neurodamus", create_tmp_simulation_config_file, "--report-buffer-size=64"],
        check=True,
        capture_output=True,
        text=True,
        env=custom_env
    )
    assert "Max Buffer size: 67108864" in result.stdout
