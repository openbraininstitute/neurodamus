"""
Tests for using POSIX shared memory /dev/shm for coreneuron_input,
controlled by CLI option --enable-shm=[ON, OFF], default: OFF
Prerequisite: an enviroment variable "SHMDIR" points to /dev/shm directory (only for linux)
"""


import os

import pytest

from tests.conftest import PLATFORM_SYSTEM

from neurodamus import Neurodamus

is_linux = PLATFORM_SYSTEM == "Linux"
# if $SHMDIR not avaible, create one for this test
if is_linux and "SHMDIR" not in os.environ:
    os.environ["SHMDIR"] = "/dev/shm"


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
def test_cli_enableshm(create_tmp_simulation_config_file, capsys):
    Neurodamus(create_tmp_simulation_config_file, enable_shm=True).run()
    captured = capsys.readouterr()

    shm_transfer_message_warning = "Unknown SHM directory for model file transfer in CoreNEURON."
    shm_transfer_message_enabled = "SHM file transfer mode for CoreNEURON enabled"

    if is_linux:
        assert shm_transfer_message_enabled in captured.out
    else:
        assert shm_transfer_message_warning in captured.out


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
def test_cli_disableshm(create_tmp_simulation_config_file, capsys):
    Neurodamus(create_tmp_simulation_config_file).run()
    captured = capsys.readouterr()

    shm_transfer_message_warning = "Unknown SHM directory for model file transfer in CoreNEURON."
    shm_transfer_message_enabled = "SHM file transfer mode for CoreNEURON enabled"

    assert shm_transfer_message_enabled not in captured.out
    assert shm_transfer_message_warning not in captured.out
