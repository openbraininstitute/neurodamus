import pytest
from pathlib import Path

pytestmark = pytest.mark.forked  # independent processes


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "sonata_config",
        "extra_config": {
            "output": {
                "log_file": "my_pydamus.log"
            }
        }
    }
], indirect=True)
def test_sonata_logfile(create_tmp_simulation_config_file):
    from neurodamus.node import Node
    # create a tmp json file to test the user defined log_file
    _ = Node(create_tmp_simulation_config_file)

    assert Path("my_pydamus.log").exists()


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "sonata_config",
        "extra_config": {
            "output": {
                "spikes_sort_order": "by_id"
            }
        }
    }
], indirect=True)
def test_throw_spike_sort_order(create_tmp_simulation_config_file):
    from neurodamus.node import Node
    from neurodamus.core.configuration import ConfigurationError

    with pytest.raises(ConfigurationError, match=r"Unsupported spikes sort order by_id"):
        _ = Node(create_tmp_simulation_config_file)
