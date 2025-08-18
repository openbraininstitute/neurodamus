import json
import subprocess
from pathlib import Path

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
