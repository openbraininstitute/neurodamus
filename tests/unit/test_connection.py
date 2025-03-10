from neurodamus import connection
import pytest

def inspect(v):
    print("-----")
    print(v, type(v))
    for i in dir(v):
        if i.startswith('_'):
            continue
        try:
            print(f"{i}: {getattr(v, i)}")
        except:
            print(f"{i}: ***")
    print("-----")

@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "simconfig_fixture": "ringtest_baseconfig",
        "extra_config": {
            "connection_overrides": [
                {
                    "name": "A2B",
                    "source": "RingA",
                    "target": "RingB",
                    "modoverride": "AMPANMDA",
                    "spont_minis": 2,
                }
            ]
        },
    },
], indirect=True)
def test_connection(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus
    from neurodamus.core import NeurodamusCore as Nd

    nd = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    conn = next(nd.circuits.get_edge_manager("RingA", "RingB").get_connections(1001))

    inspect(conn.spont_minis)

    assert False

    # conn = connection.Connection(sgid=0, tgid=0)

    # # test properties and NotImplementedErrors
    # assert conn.synapse_params is None
    # assert conn.synapses == ()
    # assert conn.synapses_offset == 0
    # assert conn.population_id == (0,0)