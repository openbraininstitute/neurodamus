import os
import struct

from neurodamus.report_parameters import ReportParameters, ReportType, SectionType, Scaling, Compartments
from neurodamus.core.configuration import _SimConfig
from neurodamus.core.coreneuron_configuration import CoreConfig
from pathlib import Path

from pytest import approx

from unittest.mock import Mock


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
