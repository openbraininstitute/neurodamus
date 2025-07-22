import os
import struct

from neurodamus.core.configuration import _SimConfig
from neurodamus.core.coreneuron_configuration import CoreConfig, CoreneuronReportConfigParameters
from pathlib import Path


def test_write_report_config(tmpdir):
    _SimConfig.output_root = str(tmpdir.join("outpath"))
    _SimConfig.coreneuron_datadir = str(tmpdir.join("datpath"))

    # Define your test parameters
    core_report_params = CoreneuronReportConfigParameters(
        report_name="soma",
        target_name="Mosaic",
        report_type="compartment",
        report_variable="v",
        unit="mV",
        report_format="SONATA",
        target_type=1,
        dt=0.1,
        start_time=0.0,
        end_time=10.0,
        gids=[1, 2, 3],
        buffer_size=8,
        scaling="none",
    )

    report_count = 1
    population_count = 20
    population_name = "default"
    population_offset = 1000
    spikes_name = "spikes.h5"
    # Call the methods with the test parameters
    CoreConfig.write_report_count(report_count)
    CoreConfig.write_report_config(core_report_params)
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
        assert parts[0] == core_report_params.report_name
        assert parts[1] == core_report_params.target_name
        assert parts[2] == core_report_params.report_type
        assert parts[3] == core_report_params.report_variable
        assert parts[4] == core_report_params.unit
        assert parts[5] == core_report_params.report_format
        assert int(parts[6]) == core_report_params.target_type
        assert float(parts[7]) == core_report_params.dt
        assert float(parts[8]) == core_report_params.start_time
        assert float(parts[9]) == core_report_params.end_time
        assert int(parts[10]) == len(core_report_params.gids)
        assert int(parts[11]) == core_report_params.buffer_size
        assert parts[12] == core_report_params.scaling
        # Read the binary data and unpack it into a list of integers
        gids_from_file = struct.unpack(f'{len(core_report_params.gids)}i', lines[2].strip())
        assert gids_from_file == tuple(core_report_params.gids), "GIDs from file do not match original GIDs"
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
