import json
import os
import numpy.testing as npt
import numpy as np


def _copy_simulation_file(src_dir, config_file, dst_dir):
    """copy simulation config file to dst_dir"""
    with open(str(src_dir / config_file)) as src_f:
        sim_config_data = json.load(src_f)
    circuit_conf = sim_config_data.get("network", "circuit_config.json")
    if not os.path.isabs(circuit_conf):
        sim_config_data["network"] = str(src_dir / circuit_conf)
    node_sets_file = sim_config_data.get("node_sets_file")
    if node_sets_file and not os.path.isabs(node_sets_file):
        sim_config_data["node_sets_file"] = str(src_dir / node_sets_file)
    with open(str(dst_dir / config_file), "w") as dst_f:
        json.dump(sim_config_data, dst_f, indent=2)
    return str(dst_dir / config_file)


def test_coreneuron_no_write_model(USECASE3, tmp_path):
    from libsonata import SpikeReader, ElementReportReader
    from neurodamus import Neurodamus
    from neurodamus.core.configuration import SimConfig

    tmp_file = _copy_simulation_file(USECASE3, "simulation_sonata_coreneuron.json", tmp_path)

    nd = Neurodamus(tmp_file, keep_build=True, coreneuron_direct_mode=True)
    nd.run()
    coreneuron_data = SimConfig.coreneuron_datadir
    assert not next(os.scandir(coreneuron_data), None), f"{coreneuron_data} should be empty."

    spikes_path = os.path.join(SimConfig.output_root, nd._run_conf.get("SpikesFile"))
    spikes_reader = SpikeReader(spikes_path)
    pop_A = spikes_reader["NodeA"]
    spike_dict = pop_A.get_dict()
    npt.assert_allclose(spike_dict["timestamps"][:10], np.array([0.2, 0.3, 0.3, 2.5, 3.4,
                                                                4.2, 5.5, 7., 7.4, 8.6]))
    npt.assert_allclose(spike_dict["node_ids"][:10], np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2]))

    soma_file = SimConfig.reports.get("soma_report").get("FileName")
    soma_path = os.path.join(SimConfig.output_root, os.path.basename(soma_file))
    soma_reader = ElementReportReader(soma_path)
    soma_A = soma_reader["NodeA"]
    soma_B = soma_reader["NodeB"]
    data_A = soma_A.get(tstop=0.5)
    data_B = soma_B.get(tstop=0.5)
    npt.assert_allclose(data_A.data, np.array([[-75.], [-39.78627], [-14.380434], [15.3370695],
                                               [1.7240616], [-13.333434]]))
    npt.assert_allclose(data_B.data, np.array([[-75.], [-75.00682], [-75.010414], [-75.0118],
                                               [-75.01173], [-75.010635]]))
