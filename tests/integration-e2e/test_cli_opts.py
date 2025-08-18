import json
import subprocess
from pathlib import Path
import pytest

import libsonata

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "v5_sonata"
CONFIG_FILE_MINI = "simulation_config_mini.json"
CIRCUIT_DIR = "sub_mini5"


def test_cli_disable_reports(tmp_path):
    with open(SIM_DIR / CONFIG_FILE_MINI, encoding="utf-8") as fd:
        sim_config_data = json.load(fd)

    sim_config_data["network"] = str(SIM_DIR / CIRCUIT_DIR / "circuit_config.json")
    with open(tmp_path / CONFIG_FILE_MINI, "w", encoding="utf-8") as fd:
        json.dump(sim_config_data, fd)

    subprocess.run(
        ["neurodamus", CONFIG_FILE_MINI, "--disable-reports"],
        check=True,
        capture_output=True,
        cwd=tmp_path
    )

    sc = libsonata.SimulationConfig.from_file(CONFIG_FILE_MINI)
    spikes_path = tmp_path / sc.output.output_dir / "out.h5"

    # Spikes are present even if we disable reports
    assert spikes_path.is_file()

    for name in sc.list_report_names:
        report_path = Path(sc.report(name).file_name)
        assert not report_path.is_file(), f"File '{report_path}' should NOT exist."

    subprocess.run(["neurodamus", CONFIG_FILE_MINI], cwd=tmp_path, check=True)
    assert spikes_path.is_file()
    for name in sc.list_report_names:
        report_path = Path(sc.report(name).file_name)
        assert report_path.is_file(), f"File '{report_path}' not found."

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
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
    tmp_path = Path(create_tmp_simulation_config_file).parent
    from os import environ
    custom_env = environ.copy()
    custom_env["SPDLOG_LEVEL"] = "debug"
    result = subprocess.run(
        ["neurodamus", create_tmp_simulation_config_file, "--report-buffer-size=64"],
        check=True,
        capture_output=True,
        text=True,
        env=custom_env
    )
    assert "Max Buffer size: 67108864" in result.stdout
