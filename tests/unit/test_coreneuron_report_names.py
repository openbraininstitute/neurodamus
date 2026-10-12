"""Reports whose names, or whose targets' names, hold whitespace, simulated with CoreNEURON.

CoreNEURON reads report.conf split on whitespace, so neurodamus writes the names there with
whitespace replaced, and renames each report's file back to its SONATA name after the run.
"""

import json
from pathlib import Path

import pytest

from ..conftest import RINGTEST_DIR
from neurodamus import Neurodamus
from neurodamus.core.configuration import ConfigurationError
from neurodamus.core.coreneuron_configuration import CoreConfig
from neurodamus.core.coreneuron_report_config import CoreReportConfig
from neurodamus.node import Node

SOMA_REPORT = {
    "type": "compartment",
    "cells": "All Rings",
    "sections": "soma",
    "variable_name": "v",
    "unit": "mV",
    "dt": 0.1,
    "start_time": 0.0,
    "end_time": 10.0,
}
LFP_REPORT = {
    "type": "lfp",
    "cells": "All Rings",
    "electrodes_file": str(RINGTEST_DIR / "lfp_file.h5"),
    "dt": 0.1,
    "start_time": 0.0,
    "end_time": 10.0,
}


def _with_node_set_holding_whitespace(config_file):
    """Add "All Rings", the ringtest's Mosaic under a name holding whitespace."""
    config_path = Path(config_file)
    config = json.loads(config_path.read_text())
    node_sets = json.loads(Path(config["node_sets_file"]).read_text())
    node_sets["All Rings"] = ["Mosaic"]
    node_sets_path = config_path.parent / "node_sets_with_whitespace.json"
    node_sets_path.write_text(json.dumps(node_sets))
    config["node_sets_file"] = str(node_sets_path)
    config_path.write_text(json.dumps(config))
    return str(config_path)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "CORENEURON",
                "reports": {"soma v": SOMA_REPORT, "lfp report": LFP_REPORT},
            },
        }
    ],
    indirect=True,
)
@pytest.mark.forked
def test_report_and_target_names_holding_whitespace(create_tmp_simulation_config_file):
    """Without the replacement, CoreNEURON aborts: Unknown string for ReportType: "All"."""
    nd = Neurodamus(_with_node_set_holding_whitespace(create_tmp_simulation_config_file))

    report_conf = CoreReportConfig.load(CoreConfig.report_config_file_save)
    assert set(report_conf.reports) == {"soma_v.h5", "lfp_report.h5"}
    assert {report.target_name for report in report_conf.reports.values()} == {"All_Rings"}

    # Unless libsonatareport is linked, CoreNEURON writes no report files, so stand-ins named
    # as it names them check that the run gives them back their SONATA names.
    output_root = Path(CoreConfig.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for file_name in ("soma_v.h5", "lfp_report.h5"):
        (output_root / file_name).write_text(file_name)

    nd.run()

    for file_name in ("soma v.h5", "lfp report.h5"):
        assert (output_root / file_name).exists()
        assert not (output_root / file_name.replace(" ", "_")).exists()


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "CORENEURON",
                "reports": {
                    "soma v": SOMA_REPORT | {"cells": "Mosaic"},
                    "soma_v": SOMA_REPORT | {"cells": "Mosaic"},
                },
            },
        }
    ],
    indirect=True,
)
def test_reports_coreneuron_would_write_to_one_file_are_refused(create_tmp_simulation_config_file):
    with pytest.raises(ConfigurationError, match="would both be written to 'soma_v.h5'"):
        Node(create_tmp_simulation_config_file)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "reports": {
                    "soma v": SOMA_REPORT | {"cells": "Mosaic"},
                    "soma_v": SOMA_REPORT | {"cells": "Mosaic"},
                },
            },
        }
    ],
    indirect=True,
)
def test_neuron_writes_them_apart(create_tmp_simulation_config_file):
    """NEURON writes its reports itself, under their SONATA names."""
    Node(create_tmp_simulation_config_file)
