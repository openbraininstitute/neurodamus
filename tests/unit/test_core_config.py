import os
import struct

from neurodamus.report_parameters import ReportParameters, ReportType, SectionType, Scaling, Compartments
from neurodamus.core.configuration import _SimConfig
from neurodamus.core.coreneuron_configuration import CoreConfig
from pathlib import Path

from pytest import approx

from unittest.mock import Mock


def test_write_report_config(tmpdir):
    _SimConfig.output_root = str(tmpdir.join("outpath"))
    _SimConfig.coreneuron_datadir = str(tmpdir.join("datpath"))

    target = Mock() # [1, 2, 3]
    target.get_gids.return_value = [1, 2, 3]
    target.name = "target_name"

    # Define your test parameters
    rep_params = ReportParameters(
        type=ReportType.COMPARTMENT,
        name="soma",
        report_on="v",
        unit="mV",
        format="SONATA",
        dt=0.1,
        start=0.0,
        end=10.0,
        output_dir=_SimConfig.output_root,
        target=target,
        buffer_size=8,
        scaling=Scaling.NONE,
        sections=SectionType.SOMA,
        compartments=Compartments.CENTER,
        compartment_set=None
    )

    report_count = 1
    population_count = 20
    population_name = "default"
    population_offset = 1000
    spikes_name = "spikes.h5"
    # Call the methods with the test parameters
    CoreConfig.write_report_count(report_count)
    CoreConfig.write_report_config(rep_params)
    CoreConfig.write_population_count(population_count)
    CoreConfig.write_spike_population(population_name, population_offset)
    CoreConfig.write_spike_filename(spikes_name)

    # Check that the report configuration file was created
    report_config_file = Path(CoreConfig.report_config_file_save)
    assert report_config_file.exists()

    # Check the content of the report configuration file
    with open(report_config_file, "rb") as fp:
        lines = fp.readlines()
        assert lines[0].strip().decode() == f"{report_count}"
        parts = lines[1].strip().decode().split()
        assert parts[0] == rep_params.name
        assert parts[1] == rep_params.target.name
        assert ReportType.from_string(parts[2]) == rep_params.type
        assert parts[3] == rep_params.report_on
        assert parts[4] == rep_params.unit
        assert parts[5] == rep_params.format
        assert SectionType.from_string(parts[6]) == rep_params.sections
        assert Compartments.from_string(parts[7]) == rep_params.compartments
        assert float(parts[8]) == approx(rep_params.dt)
        assert float(parts[9]) == approx(rep_params.start)
        assert float(parts[10]) == approx(rep_params.end)
        assert int(parts[11]) == len(rep_params.target.get_gids())
        assert int(parts[12]) == rep_params.buffer_size
        assert Scaling.from_string(parts[13]) == rep_params.scaling
        # Read the binary data and unpack it into a list of integers
        gids_from_file = struct.unpack(f'{len(rep_params.target.get_gids())}i', lines[2].strip())
        assert gids_from_file == tuple(rep_params.target.get_gids()), "GIDs from file do not match original GIDs"
        assert lines[3].strip().decode() == f"{population_count}"
        assert lines[4].strip().decode() == f"{population_name} {population_offset}"
        assert lines[5].strip().decode() == f"{spikes_name}"


def test_write_sim_config(tmpdir):
    _SimConfig.output_root = str(tmpdir.join("outpath"))
    _SimConfig.coreneuron_datadir = str(tmpdir.join("datpath"))
    cell_permute = 0
    tstop = 100
    dt = 0.1
    prcellgid = 0
    seed = 12345
    celsius = 34.0
    v_init = -65.0
    model_stats = True
    pattern = "file_pattern"
    enable_reports = 1
    report_conf = f"{CoreConfig.report_config_file_save}"
    CoreConfig.write_sim_config(
        tstop,
        dt,
        prcellgid,
        celsius,
        v_init,
        pattern,
        seed,
        model_stats,
        enable_reports
    )
    # Check that the sim configuration file was created
    sim_config_file = Path(CoreConfig.output_root) / CoreConfig.sim_config_file
    assert sim_config_file.exists()
    # Check the content of the simulation configuration file
    with open(sim_config_file, "r") as fp:
        lines = fp.readlines()
        assert lines[0].strip() == f"outpath='{Path(CoreConfig.output_root).absolute()}'"
        assert lines[1].strip() == f"datpath='{Path(CoreConfig.datadir).absolute()}'"
        assert lines[2].strip() == f"tstop={tstop}"
        assert lines[3].strip() == f"dt={dt}"
        assert lines[4].strip() == f"prcellgid={prcellgid}"
        assert lines[5].strip() == f"celsius={celsius}"
        assert lines[6].strip() == f"voltage={v_init}"
        assert lines[7].strip() == f"cell-permute={cell_permute}"
        assert lines[8].strip() == f"pattern='{pattern}'"
        assert lines[9].strip() == f"seed={seed}"
        assert lines[10].strip() == "'model-stats'"
        assert lines[11].strip() == f"report-conf='{report_conf}'"
        assert lines[12].strip() == f"mpi={os.environ.get('NEURON_INIT_MPI', '1')}"
