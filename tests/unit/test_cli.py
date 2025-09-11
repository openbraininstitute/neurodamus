import subprocess
from pathlib import Path

import libsonata
import pytest

from neurodamus.core.coreneuron_report_config import CoreReportConfig
from neurodamus.core.coreneuron_simulation_config import CoreSimulationConfig

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig"}],
    indirect=True,
)
def test_cli_color(create_tmp_simulation_config_file):
    out = subprocess.run(
               ["neurodamus", create_tmp_simulation_config_file],
               check=False,
               capture_output=True,
               cwd=str(Path(create_tmp_simulation_config_file).parent)
               )
    assert b"\033[0m" in out.stdout

    out = subprocess.run(
               ["neurodamus", create_tmp_simulation_config_file, "--use-color=OFF"],
               check=False,
               capture_output=True,
               cwd=str(Path(create_tmp_simulation_config_file).parent)
               )
    assert b"\033[0m" not in out.stdout


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig"}],
    indirect=True,
)
def test_cli_prcellgid(create_tmp_simulation_config_file):
    tmp_path = Path(create_tmp_simulation_config_file).parent
    subprocess.run(
        ["neurodamus", create_tmp_simulation_config_file, "--dump-cell-state=1", "--keep-build"],
        check=True,
    )
    assert (tmp_path / "output" / "2_py_Neuron_t0.0.nrndat").is_file()
    assert (tmp_path / "output" / "2_py_Neuron_t50.0.nrndat").is_file()


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {"target_simulator": "CORENEURON"},
        }
    ],
    indirect=True,
)
def test_cli_keep_build(create_tmp_simulation_config_file):
    tmp_path = Path(create_tmp_simulation_config_file).parent
    subprocess.run(["neurodamus", create_tmp_simulation_config_file, "--keep-build"], check=True)
    coreneuron_input_dir = tmp_path / "build" / "coreneuron_input"
    assert coreneuron_input_dir.is_dir(), "Directory 'coreneuron_input' not found."


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {"target_simulator": "CORENEURON"},
        }
    ],
    indirect=True,
)
def test_cli_build_model(create_tmp_simulation_config_file):
    res = subprocess.run(
        [
            "neurodamus",
            create_tmp_simulation_config_file,
            "--simulate-model=OFF",
            "--disable-reports",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[SKIPPED] SIMULATION (MODEL BUILD ONLY)" in res.stdout

    res = subprocess.run(
        ["neurodamus", create_tmp_simulation_config_file, "--disable-reports"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SIMULATION (SKIP MODEL BUILD)" in res.stdout

    # run it once to create the data, so that the --build-model=OFF works
    subprocess.run(
        [
            "neurodamus",
            create_tmp_simulation_config_file,
            "--simulate-model=OFF",
            "--disable-reports",
        ],
        check=True,
    )
    res = subprocess.run(
        ["neurodamus", create_tmp_simulation_config_file, "--build-model=OFF", "--disable-reports"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SIMULATION (SKIP MODEL BUILD)" in res.stdout


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig"}],
    indirect=True,
)
def test_cli_lb_mode(create_tmp_simulation_config_file):
    tmp_path = Path(create_tmp_simulation_config_file).parent
    for lb_mode in ("WholeCell", "MultiSplit"):
        res = subprocess.run(
            [
                "neurodamus",
                create_tmp_simulation_config_file,
                f"--lb-mode={lb_mode}",
                "--disable-reports",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert f"Load Balancing ENABLED. Mode: {lb_mode}" in res.stdout
        assert (tmp_path / "mcomplex.dat").is_file(), "File 'mcomplex.dat' not found."
        assert (tmp_path / "sim_conf").is_dir(), "Directory 'sim_conf' not found."


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig"}],
    indirect=True,
)
def test_cli_output_path(create_tmp_simulation_config_file):
    tmp_path = Path(create_tmp_simulation_config_file).parent
    new_output = "new-output"
    subprocess.run(
        ["neurodamus", create_tmp_simulation_config_file, f"--output-path={new_output}"],
        check=True,
        text=True,
    )
    sc = libsonata.SimulationConfig.from_file(create_tmp_simulation_config_file)
    # Output directory from simulation configuration is overridden
    simconfig_output_path = tmp_path / sc.output.output_dir
    assert not simconfig_output_path.is_dir(), (
        f"Directory '{simconfig_output_path}' should NOT exist."
    )
    assert (tmp_path / new_output).is_dir(), f"Directory '{new_output}' not found."


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "target_simulator": "CORENEURON",
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
], indirect=True)
def test_cli_report_buff_size(create_tmp_simulation_config_file):
    command = ["neurodamus", create_tmp_simulation_config_file, "--report-buffer-size=64", "--keep-build"]
    subprocess.run(command, check=True, capture_output=True)

    report_confs = CoreReportConfig.load("build/report.conf")
    assert report_confs.reports["soma_v.h5"].buffer_size == 64


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig"}],
    indirect=True,
)              
def test_cli_report_buff_invalid(create_tmp_simulation_config_file):
    for value in [-64, 0]:
        command = ["neurodamus", create_tmp_simulation_config_file, f"--report-buffer-size={value}"]
        result = subprocess.run(command, check=False, capture_output=True, text=True)

        assert "Report buffer size must be > 0" in result.stdout

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig",
              "extra_config": {
            "target_simulator": "CORENEURON",
        }
      }],
    indirect=True,
)
def test_cli_cell_permute_simple_setting(create_tmp_simulation_config_file):
    command = ["neurodamus", create_tmp_simulation_config_file, "--cell-permute=node-adjacency", "--keep-build"]
    subprocess.run(command, check=False, capture_output=True, text=True)
    sim_conf = CoreSimulationConfig.load("build/sim.conf")
    assert sim_conf.cell_permute == 1

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig",
              "extra_config": {
            "target_simulator": "CORENEURON",
        }
      }],
    indirect=True,
)
def test_cli_cell_permute_default(create_tmp_simulation_config_file):
    command = ["neurodamus", create_tmp_simulation_config_file, "--keep-build"]
    subprocess.run(command, check=False, capture_output=True, text=True)
    sim_conf = CoreSimulationConfig.load("build/sim.conf")
    assert sim_conf.cell_permute == 0

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig",
              "extra_config": {
            "target_simulator": "CORENEURON",
        }
      }],
    indirect=True,
)            
def test_cli_cell_permute_invalid(create_tmp_simulation_config_file):
    command = ["neurodamus", create_tmp_simulation_config_file, "--cell-permute=2"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert "'2' is not a valid CellPermute" in result.stdout