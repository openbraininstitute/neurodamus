
from pathlib import Path

import libsonata
import numpy.testing as npt
import pytest

from neurodamus import Neurodamus
from neurodamus.core.configuration import SimConfig
from ..conftest import V5_SONATA, RINGTEST_DIR
from ..utils import ReportReader
import copy
from neurodamus.utils.pyutils import CumulativeError
from neurodamus.core.coreneuron_simulation_config import CoreSimulationConfig


_BASE_EXTRA_CONFIG = {
            "simconfig_fixture": "TO_BE_REPLACED",
            "extra_config": {
                "target_simulator": "TO_BE_REPLACED",
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
                        "node_set": "TO_BE_REPLACED",
                    },
                },
                "reports": {
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
def make_extra_config(base, simulator):
    assert base in ["v5_sonata_config", "ringtest_baseconfig"], f"Unsupported base config: {base}"

    ans = copy.deepcopy(_BASE_EXTRA_CONFIG)
    ans["extra_config"]["target_simulator"] = simulator
    ans["simconfig_fixture"] = base

    if base == "v5_sonata_config":
        ans["extra_config"]["inputs"]["Stimulus"]["node_set"] = "Mini5"
        ans["extra_config"]["compartment_sets_file"] = str(V5_SONATA / "compartment_sets.json")
        ans["extra_config"]["reports"]["compartment_set_v"] = {
            "type": "compartment_set",
            "compartment_set": "cs1",
            "variable_name": "v",
            "dt": 1,
            "start_time": 0.0,
            "end_time": 40.0,
            "scaling": "none"
        }
        ans["extra_config"]["reports"]["compartment_set_i_membrane"] = {
            "type": "compartment_set",
            "compartment_set": "cs1",
            "variable_name": "i_membrane",
            "dt": 1,
            "start_time": 0.0,
            "end_time": 40.0,
            "scaling": "none"
        }
        ans["extra_config"]["reports"]["compartment_set_pas"] = {
            "type": "compartment_set",
            "compartment_set": "cs1",
            "variable_name": "pas",
            "dt": 1,
            "start_time": 0.0,
            "end_time": 40.0,
            "scaling": "none"
        }
        ans["extra_config"]["reports"]["synapse_ProbAMPANMDA_EMS_tau_d_AMPA"] = {
                "type": "synapse",
                "cells": "Mosaic",
                "variable_name": "ProbAMPANMDA_EMS.tau_d_AMPA",
                "sections": "all",
                "unit": "nS",
                "dt": 1,
                "start_time": 0.0,
                "end_time": 40.0,
        }
    else:
        ans["extra_config"]["inputs"]["Stimulus"]["node_set"] = "RingA"
        ans["extra_config"]["compartment_sets_file"] = str(RINGTEST_DIR / "compartment_sets.json")
        ans["extra_config"]["reports"]["compartment_set_A_v"] = {
            "type": "compartment_set",
            "compartment_set": "csA",
            "variable_name": "v",
            "dt": 1,
            "start_time": 0.0,
            "end_time": 40.0,
            "scaling": "none"
        }
        ans["extra_config"]["reports"]["compartment_set_B_v"] = {
            "type": "compartment_set",
            "compartment_set": "csB",
            "variable_name": "v",
            "dt": 1,
            "start_time": 0.0,
            "end_time": 40.0,
            "scaling": "none"
        }

    return ans


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        make_extra_config("v5_sonata_config", "NEURON"),
        make_extra_config("v5_sonata_config", "CORENEURON"),
        make_extra_config("ringtest_baseconfig", "NEURON"),
        make_extra_config("ringtest_baseconfig", "CORENEURON")
    ],
    indirect=True,
)
@pytest.mark.slow
def test_reports_compartment_vs_summation_reference_compartment_set(create_tmp_simulation_config_file):
    """
    Test that the summation report matches the summed compartment report.

    Runs a simulation generating both compartment and summation reports for 'pas',
    then asserts that summing compartment data per gid equals the summation report data,
    within numerical tolerance.
    """
    nd = Neurodamus(create_tmp_simulation_config_file)
    output_dir = Path(SimConfig.output_root)
    is_v5_sonata = "output_sonata2" in str(output_dir)
    reference_dir = V5_SONATA / "reference" / "reports" if is_v5_sonata else RINGTEST_DIR / "reference" / "reports"

    nd.run()
    loose_tols = {"rtol": 1e-6, "atol": 1e-6}

    # compartment vs summation
    for var in ["v", "i_membrane", "pas"]:
        r_compartment = ReportReader(output_dir / f"compartment_{var}.h5")
        r_summation = ReportReader(output_dir / f"summation_{var}.h5")

        r_compartment.convert_to_summation()
        # summation after report printing loses accuracy due to truncation
        # We use the loose tols in that case
        assert r_compartment.allclose(r_summation, **loose_tols), f"The summation-converted-compartment:\n{r_compartment}\ndiffers from the summation one:\n{r_summation}"

    # summation vs summation. Variable reordering
    r_summation_i_membrane_IClamp = ReportReader(output_dir / f"summation_i_membrane_IClamp.h5")
    r_summation_IClamp_i_membrane = ReportReader(output_dir / f"summation_IClamp_i_membrane.h5")
    assert r_summation_i_membrane_IClamp == r_summation_IClamp_i_membrane, (
        "Reports from 'summation_i_membrane_IClamp.h5' and 'summation_IClamp_i_membrane.h5' differ."
    )

    # summation vs manual sum
    r_summation_IClamp = ReportReader(output_dir / f"summation_IClamp.h5")
    r_compartment_i_membrane = ReportReader(output_dir / f"compartment_i_membrane.h5")
    r_summation_i_membrane_IClamp_manual = r_compartment_i_membrane+ r_summation_IClamp
    # summation after report printing loses accuracy due to truncation
    # We use the loose tols in that case
    assert r_summation_i_membrane_IClamp.allclose(r_summation_i_membrane_IClamp_manual, **loose_tols), "Summation report does not match manual addition of compartment_i_membrane and summation_IClamp reports."

    # Compare files to reference. Since the reference is fixed, this is also a comparison neuron vs coreneuron
    # reference produced with neuron
    # coreneuron does not have exactly the same results, we use the loose tols in that case
    loose_tol_files = {"summation_i_membrane.h5"}
    for ref_file in reference_dir.glob("*.h5"):
        r_reference = ReportReader(ref_file)   
        file = output_dir / ref_file.name 
        r = ReportReader(file)

        assert r.allclose(r_reference, **(loose_tols if ref_file.name in loose_tol_files else {})), f"The reports differ:\n{file}\n{ref_file}"

    # compartment vs compartment_set
    # magic list of positions in the full compartment list. It was done by hand because there isn't a clear cut way
    # to associate columns among compartment and compartment_sets. In particular there is no compartment_id in the 
    # reports (nor offset)
    if is_v5_sonata:
        ids = [0, 7, 7, 8, 190, 206, 348, 360]
        for var in ["v", "i_membrane", "pas"]:
            r_compartment = ReportReader(output_dir / f"compartment_{var}.h5")
            r_compartment = r_compartment.reduce_to_compartment_set_report("default", ids)
            r_compartment_set = ReportReader(output_dir / f"compartment_set_{var}.h5")

            assert r_compartment == r_compartment_set, f"Compartment and compartment_set reports differ for var: `{var}`\n{r_compartment}\r{r_compartment_set}"
    else:
        r_compartment = ReportReader(output_dir / "compartment_v.h5")
        r_compartment_A = r_compartment.reduce_to_compartment_set_report("RingA", [5, 8, 9])
        r_compartment_set_A = ReportReader(output_dir / "compartment_set_A_v.h5")
        assert r_compartment_A == r_compartment_set_A
        r_compartment_B = r_compartment.reduce_to_compartment_set_report("RingB", [0, 1, 2])
        r_compartment_set_B = ReportReader(output_dir / "compartment_set_B_v.h5")
        assert r_compartment_B == r_compartment_set_B

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
{
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "reports": {
                    "compartment_IClamp": {
                        "type": "compartment",
                        "cells": "Mosaic",
                        "variable_name": "IClamp",
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
def test_compartment_missing_ref(create_tmp_simulation_config_file):
    """
    Compartment reports should raise an error when requesting a reference value 
    that is not present in all compartments.
    """
    with pytest.raises(CumulativeError, match="No reference found for variable 'i' of mechanism 'IClamp' at location 0.5"): 
        Neurodamus(create_tmp_simulation_config_file)

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
{
            "simconfig_fixture": "v5_sonata_config",
            "extra_config": {
                "target_simulator": "NEURON",
                "reports": {
                    "compartment_ProbAMPANMDA_EMS": {
                        "type": "compartment",
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
def test_compartment_missing_ref(create_tmp_simulation_config_file):
    """
    Compartment reports should raise an error when requesting a reference value 
    that is not present in all compartments.
    """
    with pytest.raises(CumulativeError, match="Expected one reference for variable 'i' of mechanism 'ProbAMPANMDA_EMS' at location 0.5, but found 8"): 
        Neurodamus(create_tmp_simulation_config_file)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
{
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "target_simulator": "NEURON",
                "compartment_sets_file": str(RINGTEST_DIR / "compartment_sets.json"),
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
                        "node_set": "RingA",
                    },
                },
                "reports": {
                    "compartment_set_A_v": {
                        "type": "compartment_set",
                        "compartment_set": "csA",
                        "variable_name": "v",
                        "dt": 1,
                        "start_time": 0.0,
                        "end_time": 40.0,
                        "scaling": "none"
                    }
                },
            },
        }

    ],
    indirect=True,
)
@pytest.mark.slow
<<<<<<< HEAD
def test_reports_compartment_set_without_cached_targets(create_tmp_simulation_config_file):
=======
def test_results_are_identical_with_single_report(create_tmp_simulation_config_file):
>>>>>>> katta/node_sets
    """
    Test that a compartment set report can be retrieved and processed
    when no other reports are present.

    Neurodamus may cache `NodesetTarget` for reuse. Since compartment sets
    depend on the general `node_sets` keyword to resolve targets, this mechanism
    is fragile. Cached targets could mask the issue. This test ensures that
    the report works correctly without relying on cached targets.
    """
    nd = Neurodamus(create_tmp_simulation_config_file)
    output_dir = Path(SimConfig.output_root)
    reference_dir = RINGTEST_DIR / "reference" / "reports"
    nd.run()

    # Compare files to reference
    file_name = "compartment_set_A_v.h5"
    r_reference = ReportReader(reference_dir / file_name)   
    r = ReportReader(output_dir / file_name )
    assert r_reference == r


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        make_extra_config("v5_sonata_config", "CORENEURON"),
    ],
    indirect=True,
)
def test_reports_cell_permute(create_tmp_simulation_config_file):
    """
    Test that enabling cell permutation (cell_permute=node-adjacency) preserves report consistency.
    """
    nd = Neurodamus(create_tmp_simulation_config_file, cell_permute="node-adjacency", keep_build=True)
    output_dir = Path(SimConfig.output_root)
    reference_dir = V5_SONATA / "reference" / "reports"
    sim_conf = CoreSimulationConfig.load("build/sim.conf")
    assert sim_conf.cell_permute == 1

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

