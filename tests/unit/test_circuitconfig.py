"""
Unit tests for the CircuitConfig class (internal data structure),
and the parsing from a circuit info
which is usually translated from a SONATA config file done in sonata_config.py
"""

import pytest

from ..conftest import RINGTEST_DIR
from neurodamus.core.configuration import ConfigurationError, make_circuit_config
from neurodamus.io.sonata_config import SonataConfig


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
        }
    ],
    indirect=True,
)
def test_ringtest_circuitconf(create_tmp_simulation_config_file):
    """
    test the usual flow: read a sonata config file and convert it to CircuitConfig
    """
    config_parser = SonataConfig(create_tmp_simulation_config_file)
    circuit_dict = config_parser.Circuit.get("RingA")
    circuit_conf = make_circuit_config(circuit_dict)
    assert circuit_conf.as_dict() == {
        "CellLibraryFile": str(RINGTEST_DIR / "nodes_A.h5"),
        "PopulationName": "RingA",
        "NodesetName": "Mosaic",
        "Engine": "METype",
        "METypePath": str(RINGTEST_DIR / "hoc"),
        "MorphologyPath": str(RINGTEST_DIR / "morphologies/asc"),
        "MorphologyType": "asc",
        "PopulationType": "biophysical",
        "nrnPath": str(RINGTEST_DIR / "local_edges_A.h5:RingA__RingA__chemical"),
    }


def test_empty_circuit():
    """
    Test manual creation of a dummy CircuitConfig structure for an empty circuit.
    the least input "CellLibraryFile": "<NONE>" is required.
    """
    circuit_dict = {"CellLibraryFile": "<NONE>"}
    circuit_conf = make_circuit_config(circuit_dict)
    assert circuit_conf.CellLibraryFile is False
    assert circuit_conf.nrnPath is False
    assert circuit_conf.MorphologyPath is False
    assert circuit_conf.as_dict() == {
        "CellLibraryFile": False,
        "nrnPath": False,
        "MorphologyPath": False,
    }


def test_dummy_edges_file():
    """
    Test manual creation of a dummy CircuitConfig structure for a circuit without edges file
    """
    circuit_dict = {
        "CellLibraryFile": str(RINGTEST_DIR / "nodes_A.h5"),
        "METypePath": str(RINGTEST_DIR / "hoc"),
        "MorphologyPath": str(RINGTEST_DIR / "morphologies/asc"),
        "nrnPath": "<NONE>",
    }
    circuit_conf = make_circuit_config(circuit_dict)
    assert circuit_conf.CellLibraryFile == str(RINGTEST_DIR / "nodes_A.h5")
    assert circuit_conf.nrnPath is False


def test_validate_morphology_path():
    """
    Test the validation of morphology path
    """
    circuit_dict = {"CellLibraryFile": str(RINGTEST_DIR / "nodes_A.h5"), "nrnPath": "<NONE>"}
    with pytest.raises(ConfigurationError, match="No morphology path provided"):
        make_circuit_config(circuit_dict)


def test_default_morphology_type():
    """
    Test the default morphology type (asc) if not given in the circuit info dict,
    and the MorphologyPath is appended with "/ascii".
    """
    circuit_dict = {
        "CellLibraryFile": str(RINGTEST_DIR / "nodes_A.h5"),
        "MorphologyPath": "dummy",
        "nrnPath": "<NONE>",
    }
    circuit_conf = make_circuit_config(circuit_dict, req_morphology=False)
    assert circuit_conf.MorphologyType == "asc"
    assert circuit_conf.MorphologyPath == "dummy/ascii"


def test_validation_file_extension():
    """
    Test the validation of file extension for CellLibraryFile and nrnPath
    """
    for circuit_dict in [
        {"CellLibraryFile": "nodes.sonata", "MorphologyPath": "dummy"},
        {"CellLibraryFile": "nodes.h5", "MorphologyPath": "dummy", "nrnPath": "edges.sonata"},
    ]:
        with pytest.raises(
            ConfigurationError,
            match=r"\*.sonata files are no longer supported, please rename them to \*.h5",
        ):
            make_circuit_config(circuit_dict)
