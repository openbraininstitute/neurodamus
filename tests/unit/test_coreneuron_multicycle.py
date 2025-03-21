import numpy as np
import pytest
from pathlib import Path


from ..conftest import SIM_DIR

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from tests.utils import check_directory

from neurodamus import Neurodamus
from neurodamus.core.configuration import SimConfig

ref_gids = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])  # 1-based
ref_timestamps = np.array([5.1, 5.1, 5.1, 25.1, 25.1, 25.1, 45.1, 45.1, 45.1])


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "CORENEURON",
                "inputs": {
                    "pulse": {
                        "module": "pulse",
                        "input_type": "current_clamp",
                        "delay": 5,
                        "duration": 50,
                        "node_set": "RingA",
                        "represents_physical_electrode": True,
                        "amp_start": 10,
                        "width": 1,
                        "frequency": 50,
                    }
                }
            }
        }
    ],
    indirect=True,
)
def test_multicycle(create_tmp_simulation_config_file):
    pass
    # """TODO
    # """
    nd = Neurodamus(create_tmp_simulation_config_file)
    nd.run()

    # AttributeError: dlsym(0x3065e5280, malloc_trim): symbol not found