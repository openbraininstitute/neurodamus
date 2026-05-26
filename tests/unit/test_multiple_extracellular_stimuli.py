from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import find_peaks

from tests.conftest import RINGTEST_DIR
from tests.utils import (
    get_expected_extracellular_potentials,
    read_ascii_report,
    record_compartment_reports,
    write_ascii_reports,
)

from neurodamus import Neurodamus
from neurodamus.core.stimuli import ElectrodeSource
from neurodamus.stimulus_manager import SpatiallyUniformEField


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1, "tstop": 20},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 5,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100, "phase": 1.570796},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_two_stimulus_blocks(create_tmp_simulation_config_file):
    """
    Two stimulus blocks, one contains a cosine field and the other contains a constant field,
    they should be summed
    1. check their stimulus managers share the same SpatiallyUniformEField instance (singleton)
    2. check the size of segment_efield_integrators, should be applied to all the segments, n_seg
    3. check ElectrodeSource.fields list contain 2 fields
    4. check potential of 1st segment should be 0 (soma),
       for 4th segment the sum of the cosine fields and constant fields
    5. check an extracellar mechanism is added to each segment
    6. check the long/unused vectors of ElectrodeSource object are cleaned at the end
    """
    from neurodamus.core import NeuronWrapper as Nd

    n = Neurodamus(create_tmp_simulation_config_file)
    assert len(n._stim_manager._stimulus) == 1
    stimulus = n._stim_manager._stimulus[0]
    assert stimulus == SpatiallyUniformEField._instance
    cellref = n.circuits.get_node_manager("RingA").get_cellref(0)
    rec_dend = Nd.Vector()
    rec_dend.record(cellref.dend[0](0.25).extracellular._ref_e)
    rec_soma = Nd.Vector()
    rec_soma.record(cellref.soma[0](0.5).extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    assert list(stimulus.stimList.keys()) == [0]
    es = stimulus.stimList[0]
    assert isinstance(es, ElectrodeSource)
    assert len(es.segment_efield_integrators) == sum(sec.nseg for sec in cellref.all)
    assert len(es.fields) == 2
    dend_efi = es.segment_efield_integrators[3]

    tot_tvec = np.concatenate([[0], np.arange(Nd.dt / 2, Nd.tstop, Nd.dt)])
    ref_soma = np.zeros(len(tot_tvec))
    ref_dend = get_expected_extracellular_potentials(tot_tvec, dend_efi, es.fields)

    npt.assert_allclose(rec_soma, ref_soma)
    npt.assert_allclose(rec_dend, ref_dend)

    assert all(sec.has_membrane("extracellular") for sec in cellref.all)
    assert es.segment_displacements is None


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1, "tstop": 30},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 5,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100, "phase": 1.570796},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 4.4,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_two_stimulus_blocks_delay(create_tmp_simulation_config_file):
    """
    1. Check the combination of two stimulus blocks, one with delay.
    3. Record one segment extracellular._ref_e and run simulation, check the values
    """
    from neurodamus.core import (
        NeuronWrapper as Nd,
    )  # Import at function level, otherwise will impact other tests

    n = Neurodamus(create_tmp_simulation_config_file)
    cellref = n.circuits.get_node_manager("RingA").get_cellref(0)
    assert len(n._stim_manager._stimulus) == 1
    stimulus = n._stim_manager._stimulus[0]
    rec_dend = Nd.Vector()
    rec_dend.record(cellref.dend[0](0.25).extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    assert list(stimulus.stimList.keys()) == [0]
    es = stimulus.stimList[0]
    assert isinstance(es, ElectrodeSource)
    assert len(es.fields) == 2

    dend_efi = es.segment_efield_integrators[3]
    tot_tvec = np.concatenate([[0], np.arange(Nd.dt / 2, Nd.tstop, Nd.dt)])
    ref_dend = get_expected_extracellular_potentials(tot_tvec, dend_efi, es.fields)

    npt.assert_allclose(rec_dend, ref_dend)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1, "tstop": 20},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 5,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100, "phase": 1.570796},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 4.4,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_3": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 4.5,
                        "duration": 6,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 200, "Ey": -100, "Ez": 100, "frequency": 0.001},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_three_stimulus_blocks_delay(create_tmp_simulation_config_file):
    """
    Check three fields in the stimlus, cosine + constant + cosine with small freq(almost constant)
    """
    from neurodamus.core import (
        NeuronWrapper as Nd,
    )  # Import at function level, otherwise will impact other tests

    n = Neurodamus(create_tmp_simulation_config_file)
    stimulus = n._stim_manager._stimulus[0]
    cellref = n.circuits.get_node_manager("RingA").get_cellref(0)
    stimulus = n._stim_manager._stimulus[0]
    rec_dend = Nd.Vector()
    rec_dend.record(cellref.dend[0](0.25).extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    es = stimulus.stimList[0]
    assert len(es.fields) == 3

    tot_tvec = np.concatenate([[0], np.arange(Nd.dt / 2, Nd.tstop, Nd.dt)])
    dend_efi = es.segment_efield_integrators[3]

    ref_dend = get_expected_extracellular_potentials(tot_tvec, dend_efi, es.fields)

    npt.assert_allclose(rec_dend, ref_dend)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 10,
                        "node_set": "RingA",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_two_blocks_nodeset_overlap(create_tmp_simulation_config_file):
    """
    Check two stimulus blocks where the node_sets overlap
    Cell 0 contains the sum of the two stimuli
    cell 1,2 contain the 2nd stimluli
    """

    from neurodamus.core import (
        NeuronWrapper as Nd,
    )  # Import at function level, otherwise will impact other tests

    n = Neurodamus(create_tmp_simulation_config_file)
    cellref0 = n.circuits.get_node_manager("RingA").get_cellref(0)
    cellref1 = n.circuits.get_node_manager("RingA").get_cellref(1)
    stimulus = n._stim_manager._stimulus[0]
    rec_dend0 = Nd.Vector()
    rec_dend0.record(cellref0.dend[0](0.25).extracellular._ref_e)
    rec_dend1 = Nd.Vector()
    rec_dend1.record(cellref1.dend[0](0.25).extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    tot_tvec = np.concatenate([[0], np.arange(Nd.dt / 2, Nd.tstop, Nd.dt)])

    # cell 0, big cell
    es0 = stimulus.stimList[0]
    assert len(es0.fields) == 2
    ref0 = get_expected_extracellular_potentials(
        tot_tvec, es0.segment_efield_integrators[3], es0.fields
    )
    npt.assert_allclose(rec_dend0, ref0)

    # cell 1, small cell
    es1 = stimulus.stimList[1]
    assert len(es1.fields) == 1
    ref1 = get_expected_extracellular_potentials(
        tot_tvec, es1.segment_efield_integrators[1], es1.fields
    )
    npt.assert_allclose(rec_dend1, ref1)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [
        {
            "simconfig_fixture": "ringtest_baseconfig",
            "extra_config": {
                "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
                "run": {"dt": 1},
                "inputs": {
                    "ex_efields_1": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 5,
                        "duration": 5,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
                        ],
                        "ramp_up_time": 3.0,
                        "ramp_down_time": 4.0,
                    },
                    "ex_efields_2": {
                        "input_type": "extracellular_stimulation",
                        "module": "spatially_uniform_e_field",
                        "delay": 0,
                        "duration": 7,
                        "node_set": "RingA_Cell0",
                        "fields": [
                            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                        ],
                    },
                },
            },
        }
    ],
    indirect=True,
)
def test_two_blocks_time_overlap(create_tmp_simulation_config_file):
    """
    Check 2 stimulus blocks with different time window
    block 1 : [5,17], block 2: [0,7]
    """
    from neurodamus.core import (
        NeuronWrapper as Nd,
    )  # Import at function level, otherwise will impact other tests

    n = Neurodamus(create_tmp_simulation_config_file)
    stimulus = n._stim_manager._stimulus[0]
    cellref = n.circuits.get_node_manager("RingA").get_cellref(0)
    stimulus = n._stim_manager._stimulus[0]
    rec_dend = Nd.Vector()
    rec_dend.record(cellref.dend[0](0.25).extracellular._ref_e)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    es = stimulus.stimList[0]
    assert len(es.fields) == 2

    tot_tvec = np.concatenate([[0], np.arange(Nd.dt / 2, Nd.tstop, Nd.dt)])
    dend_efi = es.segment_efield_integrators[3]

    ref = get_expected_extracellular_potentials(tot_tvec, dend_efi, es.fields)
    npt.assert_allclose(rec_dend, ref)


@pytest.mark.parametrize(
    ("create_tmp_simulation_config_file", "ref_peak"),
    [
        (
            {
                "simconfig_fixture": "ringtest_baseconfig",
                "extra_config": {
                    "network": str(RINGTEST_DIR / "circuit_config_bigA.json"),
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
                        },
                        "ex_efields_1": {
                            "input_type": "extracellular_stimulation",
                            "module": "spatially_uniform_e_field",
                            "delay": 0,
                            "duration": 50,
                            "node_set": "RingA_Cell0",
                            "fields": [
                                {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 10},
                            ],
                            "ramp_up_time": 1.0,
                            "ramp_down_time": 2.0,
                        },
                        "ex_efields_2": {
                            "input_type": "extracellular_stimulation",
                            "module": "spatially_uniform_e_field",
                            "delay": 0,
                            "duration": 50,
                            "node_set": "RingA_Cell0",
                            "fields": [
                                {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
                            ],
                            "ramp_up_time": 1.0,
                            "ramp_down_time": 2.0,
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
                            "end_time": 20.0,
                        }
                    },
                },
            },
            [
                6,
                21,
                27,
                42,
                48,
                63,
                69,
                90,
                111,
                132,
                153,
                174,
                195,
                216,
                237,
                258,
                279,
                300,
                321,
                342,
                363,
                384,
                405,
                426,
                447,
                468,
            ],
        ),
    ],
    indirect=["create_tmp_simulation_config_file"],
)
def test_neuron_report_with_efields(create_tmp_simulation_config_file, ref_peak):
    """NEURON integration tests, check the compartment ASCII report"""
    from neurodamus.core import (
        NeuronWrapper as Nd,
    )  # Import at function level, otherwise will impact other tests

    n = Neurodamus(create_tmp_simulation_config_file)
    ascii_recorders = record_compartment_reports(n._target_manager)
    Nd.finitialize()  # reinit for the recordings to be registered
    n.run()

    # Write ASCII reports
    write_ascii_reports(ascii_recorders, n._run_conf.output_root)

    # Read ASCII reports
    report = Path(n._run_conf.output_root) / ("compartment_v.txt")
    assert report.exists()
    data = read_ascii_report(report)
    cell0_voltage_vec = [vec[3] for vec in data if vec[0] == 0]
    assert len(cell0_voltage_vec) == 21 * 23  # 21 time steps * 23 compartments
    peaks_pos = find_peaks(cell0_voltage_vec, prominence=1)[0]
    npt.assert_allclose(peaks_pos, ref_peak)
    n.clear_model()  # clear up the reporting vector, required for the next run.
