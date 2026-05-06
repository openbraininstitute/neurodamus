from pathlib import Path

import pytest
from libsonata import SonataError

from neurodamus.core.configuration import SimConfig
from neurodamus.core.coreneuron_configuration import CoreConfig
from neurodamus.utils.pyutils import CumulativeError
from neurodamus.node import Node
from tests.utils import (read_ascii_report,
                         record_compartment_reports, write_ascii_reports)

import numpy as np
from scipy.signal import find_peaks



@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "reports": {
                    "wrong_variable": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "wrong",
                        "sections": "all",
                        "dt": 0.1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                }
            },
        },
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "reports": {
                    "too_small_dt": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "wrong",
                        "sections": "all",
                        "dt": 0.01,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                }
            },
        },
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "reports": {
                    "wrong_start_end_time": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "wrong",
                        "sections": "all",
                        "dt": 0.01,
                        "start_time": 30.0,
                        "end_time": 0.0,
                    }
                }
            },
        },
    ],
    indirect=True,
)
def test_config_CumulativeError(create_tmp_simulation_config_file):
    """Test error handling in enable_reports:
    1. wrong variable name
    2. dt < simulation dt
    3. start_time > end_time
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    with pytest.raises(CumulativeError, match="is before start time|is smaller than simulation dt|reference found for variable 'i' of mechanism 'wrong'|reports requires exactly one variable, but received"):
        n.enable_reports()

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "reports": {
                    "report": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "sections": "all",
                        "compartments": "all",
                        "variable_name": "i_membrane,IClamp",
                        "unit": "nA",
                        "dt": 10,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_config_addional_errors(create_tmp_simulation_config_file):
    """Test error handling in enable_reports:
    1. wrong variable name
    2. dt < simulation dt
    3. start_time > end_time
    """
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    with pytest.raises(CumulativeError, match="reports requires exactly one variable, but received"):
        n.enable_reports()

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "target_simulator": "CORENEURON",
            "extra_config": {
                "reports": {
                    "wrong_variable": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "v",
                        "sections": "abc",
                        "dt": 0.1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                }
            },
        }
    ],
    indirect=True,
)
def test_wrong_report_sections(create_tmp_simulation_config_file):
    """Should be caught by libsonata parser"""
    with pytest.raises(SonataError, match=r"Invalid value: \'\"abc\"\' for key \'sections\'"):
        Node(create_tmp_simulation_config_file)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "target_simulator": "CORENEURON",
            "extra_config": {
                "reports": {
                    "wrong_variable": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "v",
                        "sections": "soma",
                        "compartments": "others",
                        "dt": 0.1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                }
            },
        }
    ],
    indirect=True,
)
def test_wrong_report_compartment(create_tmp_simulation_config_file):
    """Should be caught by libsonata parser"""
    with pytest.raises(
        SonataError, match=r"Invalid value: \'\"others\"\' for key \'compartments\'"
    ):
        Node(create_tmp_simulation_config_file)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "CORENEURON",
                "reports": {
                    "synapse_report": {
                        "type": "synapse",
                        "cells": "RingA",
                        "sections": "all",
                        "variable_name": "ProbAMPANMDA_EMS.g",
                        "unit": "nA",
                        "dt": 10,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_enable_synapse_report_errorhandling(create_tmp_simulation_config_file):
    """Syanpse report is not possible with the ringtest circuit, lack of synapses in cell"""
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    with pytest.raises(CumulativeError, match=r"'ProbAMPANMDA_EMS' not found"):
        n.enable_reports()


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "reports": {
                    "new_report": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "wrong",
                        "sections": "all",
                        "dt": 0.1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "enabled": False,
                    }
                }
            },
        }
    ],
    indirect=True,
)
def test_report_disabled(create_tmp_simulation_config_file):
    n = Node(create_tmp_simulation_config_file)
    n.load_targets()
    n.create_cells()
    n.enable_reports()
    assert len(n.reports) == 0

@pytest.mark.slow
@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "inputs": {
                    "Stimulus": {
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
                },
                "reports": {
                    "soma_v": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "v",
                        "sections": "soma",
                        "dt": 0.1,
                        "start_time": 0.0,
                        "end_time": 50.0,
                    },
                    "compartment_i": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "i_membrane",
                        "sections": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    },
                    "compartment_pas": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "pas",
                        "sections": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_neuron_compartment_ASCIIReport(create_tmp_simulation_config_file):
    """For unit tests, we don't build libsonatareport to create the standard sonata reports,
    instead we use custom functions to record and write report vectors in ASCII format,
    but currently only for compartment reports
    """
    from neurodamus import Neurodamus
    from neurodamus.core import NeuronWrapper as Nd
    from neurodamus.core.configuration import SimConfig

    n = Neurodamus(create_tmp_simulation_config_file)
    assert len(n.reports) == 3
    ascii_recorders = record_compartment_reports(n._target_manager)

    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    # Write ASCII reports
    write_ascii_reports(ascii_recorders, n._run_conf.output_root)

    # Read ASCII reports
    soma_report = Path(n._run_conf.output_root) / ("soma_v.txt")
    assert soma_report.exists()
    data = read_ascii_report(soma_report)
    assert len(data) == 2500  # 500 time steps * 5 soma sections
    # check soma signal peak for cell 1001 as in test_current_injection.py
    cell_voltage_vec = [vec[3] for vec in data if vec[0] == 1000]
    peaks_pos = find_peaks(cell_voltage_vec, prominence=1)[0]
    np.testing.assert_allclose(peaks_pos, [92, 291])

    compartment_i_report = Path(n._run_conf.output_root) / ("compartment_i.txt")
    assert compartment_i_report.exists()
    data = read_ascii_report(compartment_i_report)
    assert len(data) == 1025  # 45 time steps * 5*5 compartments
    cell_current_vec = [vec[3] for vec in data if vec[0] == 1000]

    peaks_pos = find_peaks(cell_current_vec, prominence=0.05)[0]
    np.testing.assert_allclose(peaks_pos, [9,  29,  50,  70, 110, 132, 152, 173, 193])
    
    compartment_pas_report = Path(n._run_conf.output_root) / ("compartment_pas.txt")
    assert compartment_pas_report.exists()
    data = read_ascii_report(compartment_pas_report)
    assert len(data) == 1025  # 45 time steps * 5*5 compartments
    cell_current_vec = [vec[3] for vec in data if vec[0] == 1000]
    assert any(cell_current_vec), "The pas current is always 0. This is very suspicious"

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "reports": {
                    "summation_report": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "sections": "soma",
                        "compartments": "center",
                        "variable_name": "i_membrane, IClamp",
                        "unit": "nA",
                        "dt": 10,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                },
            },
        },
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "reports": {
                    "summation_report": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "sections": "all",
                        "compartments": "all",
                        "variable_name": "i_membrane, IClamp",
                        "unit": "nA",
                        "dt": 10,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                },
            },
        },
    ],
    indirect=True,
)
def test_enable_summation_report(create_tmp_simulation_config_file):
    """Check summartion report is enabled in different cases:
    1. Neuron, sum_currents_into_soma = True(sections=soma, compartments=center)
    2. Neuron, sum_currents_into_soma = False(sections=all, compartments=all)
    """
    from neurodamus import Neurodamus

    n = Neurodamus(create_tmp_simulation_config_file)
    assert len(n.reports) == 1
    assert n.reports[0].variables == [("i_membrane_", "i"), ("IClamp", "i")]

    if SimConfig.use_coreneuron:
        assert Path(CoreConfig.report_config_file_save).exists()


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "CORENEURON",
                "reports": {
                    "soma": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "sections": "all",
                        "variable_name": "v",
                        "unit": "nA",
                        "dt": 10,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                },
            },
        },
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "CORENEURON",
                "reports": {
                    "summation": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "sections": "soma",
                        "variable_name": "i_membrane, IClamp",
                        "unit": "nA",
                        "dt": 10,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    }
                },
            },
        }
    ],
    indirect=True,
)
def test_enable_coreneuron_report(create_tmp_simulation_config_file):
    """Check report is enabled for CoreNEURON and report.conf is created
    1. compartment report, sections = soma
    2. summation report, sections = all
    """
    from neurodamus import Neurodamus

    n = Neurodamus(create_tmp_simulation_config_file)
    assert len(n.reports) == 1
    assert Path(CoreConfig.report_config_file_save).exists()
