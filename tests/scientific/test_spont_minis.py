import numpy
import pytest

SPONT_RATE = 100


@pytest.mark.slow
@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "sonata_config",
        "extra_config": {
            "run": {
                "random_seed": 12345,
                "dt": 0.05,
                "tstop": 20,
            },
            "connection_overrides": [
                {
                    "name": "in-nodeA",
                    "source": "nodesPopA",
                    "target": "l4pc",
                    "spont_minis": SPONT_RATE,
                    "synapse_configure": "%s.verboseLevel=1"  # output when a spike is received
                }
            ]
        }
    }
], indirect=True)
def test_spont_minis(create_tmp_simulation_config_file):
    from neurodamus.connection_manager import Nd, SynapseRuleManager
    from neurodamus import Neurodamus
    from neurodamus.core.configuration import Feature

    nd = Neurodamus(
        create_tmp_simulation_config_file,
        restrict_node_populations=["NodeA"],
        restrict_features=[Feature.SpontMinis],  # Enable Feature.SynConfigure to see events
        restrict_connectivity=1,  # base restriction, no projections
        disable_reports=True,
    )

    edges_a: SynapseRuleManager = nd.circuits.get_edge_manager("NodeA", "NodeA")
    # Note: Before #198 we would instantiate a projection conn as being internal
    assert len(list(edges_a.all_connections())) == 1
    conn_1_0 = next(edges_a.get_connections(0, 1))
    assert conn_1_0._spont_minis.rate == SPONT_RATE

    c0 = edges_a.cell_manager.get_cellref(0)
    voltage_vec = Nd.Vector()
    voltage_vec.record(c0.soma[0](0.5)._ref_v)
    Nd.finitialize()  # reinit for the recordings to be registered

    nd.run()

    # When we get an event the voltage drops
    # We find that looking at the acceleration of the voltage drop
    # We do a convolution to weight in neighbor points and have a smoother line
    v_increase_rate = numpy.diff(voltage_vec, 2)
    window_sum = numpy.convolve(v_increase_rate, [1, 2, 4, 2, 1], 'valid')
    # print(numpy.array_str(window_sum, suppress_small=True))
    strong_reduction_pos = numpy.nonzero(window_sum < -0.01)[0]
    # At least one such point, at most 2% of all points
    assert 1 <= len(strong_reduction_pos) <= int(0.02 * len(window_sum))
