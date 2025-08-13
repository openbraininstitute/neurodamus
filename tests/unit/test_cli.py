from pathlib import Path
import subprocess
import sys
from unittest import mock
import pytest

from neurodamus import Neurodamus


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
