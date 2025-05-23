import json
import os
import subprocess
import tempfile
from pathlib import Path

SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations" / "v5_sonata"
CONFIG_FILE_MINI = "simulation_config_mini.json"
CIRCUIT_DIR = "sub_mini5"


def test_cli_prcellgid():
    from neurodamus import Neurodamus
    test_folder = tempfile.TemporaryDirectory("cli-test-prcellgid")  # auto removed
    test_folder_path = Path(test_folder.name)
    with open(SIM_DIR / CONFIG_FILE_MINI, "r") as f:
        sim_config_data = json.load(f)
        sim_config_data["network"] = str(SIM_DIR / CIRCUIT_DIR / "circuit_config.json")
        with open(test_folder_path / CONFIG_FILE_MINI, "w") as f:
            json.dump(sim_config_data, f, indent=2)

    os.chdir(test_folder_path)
    nd = Neurodamus(CONFIG_FILE_MINI, dump_cell_state=1, keep_build=True
                    )
    nd.run()
    assert (test_folder_path / "output_sonata2" / "2_py_Neuron_t0.0.nrndat").is_file()
    assert (test_folder_path / "output_sonata2"  / "2_py_Neuron_t100.0.nrndat").is_file()


def test_cli_disable_reports():
    test_folder = tempfile.TemporaryDirectory("cli-test-disable-reports")  # auto removed
    test_folder_path = Path(test_folder.name)
    with open(SIM_DIR / CONFIG_FILE_MINI, "r") as f:
        sim_config_data = json.load(f)
        sim_config_data["network"] = str(SIM_DIR / CIRCUIT_DIR / "circuit_config.json")
        with open(test_folder_path / CONFIG_FILE_MINI, "w") as f:
            json.dump(sim_config_data, f, indent=2)

    subprocess.run(
        ["neurodamus", CONFIG_FILE_MINI, "--disable-reports"],
        check=True,
        capture_output=True,
        cwd=test_folder_path
    )
    # Spikes are present even if we disable reports
    assert (test_folder_path / sim_config_data["output"]["output_dir"] / "out.h5").is_file()
    for report in sim_config_data["reports"].keys():
        report_path = test_folder_path / sim_config_data["output"]["output_dir"] / (report + ".h5")
        assert not report_path.is_file(), f"File '{report_path}' should NOT exist."

    subprocess.run(
        ["neurodamus", CONFIG_FILE_MINI],
        check=True,
        capture_output=True,
        cwd=test_folder_path
    )
    assert (test_folder_path / sim_config_data["output"]["output_dir"] / "out.h5").is_file()
    for report in sim_config_data["reports"].keys():
        report_path = test_folder_path / sim_config_data["output"]["output_dir"] / (report + ".h5")
        assert report_path.is_file(), f"File '{report_path}' not found."


def test_cli_keep_build():
    from neurodamus import Neurodamus
    with open(SIM_DIR / CONFIG_FILE_MINI, "r") as f:
        sim_config_data = json.load(f)
        sim_config_data["target_simulator"] = "CORENEURON"
        sim_config_data["output"]["output_dir"] = "output_keep_build"
        sim_config_data["network"] = str(SIM_DIR / CIRCUIT_DIR / "circuit_config.json")

    test_folder = tempfile.TemporaryDirectory("cli-test-keep-build")  # auto removed
    test_folder_path = Path(test_folder.name)
    with open(test_folder_path / CONFIG_FILE_MINI, "w") as f:
        json.dump(sim_config_data, f, indent=2)

    os.chdir(test_folder_path)
    nd = Neurodamus(CONFIG_FILE_MINI, keep_build=True, disable_reports=True)
    nd.run()
    coreneuron_input_dir = test_folder_path / "build" / "coreneuron_input"
    assert coreneuron_input_dir.is_dir(), "Directory 'coreneuron_input' not found."


def test_cli_build_model():
    with open(SIM_DIR / CONFIG_FILE_MINI, "r") as f:
        sim_config_data = json.load(f)
        sim_config_data["target_simulator"] = "CORENEURON"
        sim_config_data["network"] = str(SIM_DIR / CIRCUIT_DIR / "circuit_config.json")

    test_folder = tempfile.TemporaryDirectory("cli-test-build-model")  # auto removed
    test_folder_path = Path(test_folder.name)
    with open(test_folder_path / CONFIG_FILE_MINI, "w") as f:
        json.dump(sim_config_data, f, indent=2)

    result_model = subprocess.run(
        ["neurodamus", CONFIG_FILE_MINI, "--simulate-model=OFF", "--disable-reports"],
        check=True,
        cwd=test_folder_path,
        capture_output=True,
        text=True
    )
    assert "[SKIPPED] SIMULATION (MODEL BUILD ONLY)" in result_model.stdout

    result_auto = subprocess.run(
        ["neurodamus", CONFIG_FILE_MINI, "--disable-reports"],
        check=True,
        cwd=test_folder_path,
        capture_output=True,
        text=True
    )
    assert "SIMULATION (SKIP MODEL BUILD)" in result_auto.stdout

    subprocess.run(
        ["neurodamus", CONFIG_FILE_MINI, "--simulate-model=OFF", "--disable-reports"],
        check=True,
        cwd=test_folder_path
    )
    result_off = subprocess.run(
        ["neurodamus", CONFIG_FILE_MINI, "--build-model=OFF", "--disable-reports"],
        check=True,
        cwd=test_folder_path,
        capture_output=True,
        text=True
    )
    assert "SIMULATION (SKIP MODEL BUILD)" in result_off.stdout


def test_cli_lb_mode():
    test_folder = tempfile.TemporaryDirectory("cli-test-lb-mode")  # auto removed
    test_folder_path = Path(test_folder.name)
    with open(SIM_DIR / CONFIG_FILE_MINI, "r") as f:
        sim_config_data = json.load(f)
        sim_config_data["network"] = str(SIM_DIR / CIRCUIT_DIR / "circuit_config.json")
        with open(test_folder_path / CONFIG_FILE_MINI, "w") as f:
            json.dump(sim_config_data, f, indent=2)

    for lb_mode in ("WholeCell", "MultiSplit"):
        result = subprocess.run(
            ["neurodamus", CONFIG_FILE_MINI, f"--lb-mode={lb_mode}", "--disable-reports"],
            check=True,
            cwd=test_folder_path,
            capture_output=True,
            text=True
        )
        assert f"Load Balancing ENABLED. Mode: {lb_mode}" in result.stdout
        assert (test_folder_path / "mcomplex.dat").is_file(), "File 'mcomplex.dat' not found."
        assert (test_folder_path / "sim_conf").is_dir(), "Directory 'sim_conf' not found."


def test_cli_output_path():
    from neurodamus import Neurodamus
    test_folder = tempfile.TemporaryDirectory("cli-test-output-path")  # auto removed
    test_folder_path = Path(test_folder.name)
    with open(SIM_DIR / CONFIG_FILE_MINI, "r") as f:
        sim_config_data = json.load(f)
        sim_config_data["network"] = str(SIM_DIR / CIRCUIT_DIR / "circuit_config.json")
        with open(test_folder_path / CONFIG_FILE_MINI, "w") as f:
            json.dump(sim_config_data, f, indent=2)

    simconfig_output_path = sim_config_data["output"]["output_dir"]
    output_path = "new_output"
    os.chdir(test_folder_path)
    nd = Neurodamus(CONFIG_FILE_MINI, output_path=output_path)
    nd.run()
    # Output directory from simulation configuration is overridden
    assert not (test_folder_path / simconfig_output_path).is_dir(), \
           f"Directory '{simconfig_output_path}' should NOT exist."
    assert (test_folder_path / output_path).is_dir(), f"Directory '{output_path}' not found."
