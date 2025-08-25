import numpy.testing as npt
import numpy as np
import pytest
from pathlib import Path


SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"

# to be enabled with #337
@pytest.mark.skip
@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(SIM_DIR / "usecase3"),
        "simconfig_file": "simulation_sonata_coreneuron.json"
    }
], indirect=True)
def test_coreneuron_no_write_model(create_tmp_simulation_config_file):
    from libsonata import SpikeReader, ElementReportReader
    from neurodamus import Neurodamus
    from neurodamus.core.coreneuron_configuration import CoreConfig
    from neurodamus.core.configuration import SimConfig

    tmp_file = create_tmp_simulation_config_file

    nd = Neurodamus(tmp_file, keep_build=True, coreneuron_direct_mode=True)
    nd.run()
    coreneuron_data = Path(CoreConfig.datadir)
    assert coreneuron_data.is_dir() and not any(coreneuron_data.iterdir()), (
        f"{coreneuron_data} should be empty."
    )

    spikes_path = Path(SimConfig.output_root) / nd._run_conf.get("SpikesFile")
    spikes_reader = SpikeReader(spikes_path)
    pop_A = spikes_reader["NodeA"]
    spike_dict = pop_A.get_dict()
    npt.assert_allclose(spike_dict["timestamps"][:10], np.array([0.2, 0.3, 0.3, 2.5, 3.4,
                                                                4.2, 5.5, 7., 7.4, 8.6]))
    npt.assert_allclose(spike_dict["node_ids"][:10], np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2]))

    soma_file = Path(SimConfig.reports["soma_report"]["FileName"]).name
    soma_path = Path(SimConfig.output_root) / soma_file
    soma_reader = ElementReportReader(soma_path)
    soma_A = soma_reader["NodeA"]
    soma_B = soma_reader["NodeB"]
    data_A = soma_A.get(tstop=0.5)
    data_B = soma_B.get(tstop=0.5)
    npt.assert_allclose(data_A.data, np.array([[-75.], [-39.78627], [-14.380434], [15.3370695],
                                               [1.7240616], [-13.333434]]))
    npt.assert_allclose(data_B.data, np.array([[-75.], [-75.00682], [-75.010414], [-75.0118],
                                               [-75.01173], [-75.010635]]))
