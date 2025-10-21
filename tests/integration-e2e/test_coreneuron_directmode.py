import numpy.testing as npt
import numpy as np
import pytest
from pathlib import Path
from neurodamus import Neurodamus
from libsonata import SpikeReader, ElementReportReader
from neurodamus.core.coreneuron_configuration import CoreConfig
from neurodamus.core.configuration import SimConfig
from ..utils import ReportReader

from ..conftest import V5_SONATA


SIM_DIR = Path(__file__).parent.parent.absolute() / "simulations"

@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(SIM_DIR / "usecase3"),
        "simconfig_file": "simulation_sonata_coreneuron.json"
    }
], indirect=True)
def test_coreneuron_no_write_model(create_tmp_simulation_config_file):

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



@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "v5_sonata_config",
            "extra_config": {
                "compartment_sets_file": str(V5_SONATA / "compartment_sets.json"),
                "target_simulator": "CORENEURON",
                "inputs": {
                    "override_field": 1,
                    "Stimulus": {
                        "module": "pulse",
                        "input_type": "current_clamp",
                        "represents_physical_electrode": True,
                        "amp_start": 3,
                        "width": 10,
                        "frequency": 50,
                        "delay": 0,
                        "duration": 50,
                        "node_set": "Mini5",
                    },
                },
                "reports": {
                    "compartment_set_i_membrane": {
                        "type": "compartment_set",
                        "compartment_set": "cs1",
                        "variable_name": "i_membrane",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none"
                    },
                    "compartment_set_pas": {
                        "type": "compartment_set",
                        "compartment_set": "cs1",
                        "variable_name": "pas",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none"
                    },
                    "synapse_ProbAMPANMDA_EMS_tau_d_AMPA": {
                            "type": "synapse",
                            "cells": "Mosaic",
                            "variable_name": "ProbAMPANMDA_EMS.tau_d_AMPA",
                            "sections": "all",
                            "unit": "nS",
                            "dt": 1,
                            "start_time": 0.0,
                            "end_time": 40.0,
                    },
                    "compartment_set_v": {
                        "type": "compartment_set",
                        "compartment_set": "cs1",
                        "variable_name": "v",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none"
                    },
                    "compartment_v": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "v",
                        "sections": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none"
                    },
                    "summation_v": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "v",
                        "sections": "soma",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
                    },
                    "compartment_i_membrane": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "i_membrane",
                        "sections": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    },
                    "summation_i_membrane": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "i_membrane",
                        "sections": "soma",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
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
                    "summation_pas": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "pas",
                        "sections": "soma",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
                    },
                    "summation_v_area_scaling": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "v",
                        "sections": "soma",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                    },
                    "summation_IClamp": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "IClamp",
                        "sections": "all",
                        "compartments": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
                    },
                    "summation_i_membrane_IClamp": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "i_membrane,IClamp",
                        "sections": "all",
                        "compartments": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
                    },
                    "summation_IClamp_i_membrane": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "IClamp,i_membrane",
                        "sections": "all",
                        "compartments": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
                    },
                    "summation_ProbAMPANMDA_EMS": {
                        "type": "summation",
                        "cells": "Mosaic",
                        "variable_name": "ProbAMPANMDA_EMS",
                        "sections": "all",
                        "compartments": "all",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none",
                    },
                },
            },
        }
    ],
    indirect=True,
)
@pytest.mark.slow
def test_reports_direct_mode_vs_reference(create_tmp_simulation_config_file):
    """
    Test coreneuron direct mode run vs reference files.
    """
    nd = Neurodamus(create_tmp_simulation_config_file, coreneuron_direct_mode=True)
    output_dir = Path(SimConfig.output_root)
    reference_dir = V5_SONATA / "reference" / "reports"

    nd.run()
    loose_tols = {"rtol": 1e-6, "atol": 1e-6}

    # Compare files to reference. Since the reference is fixed, this is also a comparison neuron vs coreneuron
    # reference produced with neuron
    # coreneuron does not have exactly the same results, we use the loose tols in that case
    loose_tol_files = {"summation_i_membrane.h5"}
    for ref_file in reference_dir.glob("*.h5"):
        r_reference = ReportReader(ref_file)   
        file = output_dir / ref_file.name 
        r = ReportReader(file)

        assert r.allclose(r_reference, **(loose_tols if ref_file.name in loose_tol_files else {})), f"The reports differ:\n{file}\n{ref_file}"