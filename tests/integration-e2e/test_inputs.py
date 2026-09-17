import json
import subprocess

import libsonata
import numpy as np
import pytest

from tests import utils

import neurodamus.core.stimuli as st


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig",
      "extra_config": {
          "inputs": {
              "Stimulus": {
                  "module": "linear",
                  "input_type": "current_clamp",
                  "delay": 0,
                  "duration": 50,
                  "node_set": "RingA",
                  "amp_start": 10,
                  "amp_end": 10,
                  "represents_physical_electrode": True,
                  }
              },
          "reports": {
              "report": {
                  "type": "compartment",
                  "cells": "RingA",
                  "variable_name": "v",
                  "sections": "soma",
                  "dt": 0.1,
                  "start_time": 0,
                  "end_time": 50,
                  }
              }
          }
      }],
    indirect=True,
)
def test_current_replay_linear(create_tmp_simulation_config_file, tmp_path):
    # run using `inputs::Stimulus`
    subprocess.run(
        ["neurodamus", create_tmp_simulation_config_file, f"--output-path={tmp_path}/input"],
        check=True,
        capture_output=True,
    )

    # run it with current input
    path = tmp_path / "linear.h5"
    ssc = libsonata.SimulationConfig.from_file(create_tmp_simulation_config_file).input("Stimulus")
    ss = st.SignalSource().add_ramp(amp1=ssc.amp_start, amp2=ssc.amp_end, duration=ssc.duration)

    times = ss.time_vec.as_numpy().astype(np.float32)
    data = ss.stim_vec.as_numpy().astype(np.float32)
    utils.write_single_compartment_report(path,
                                          times,
                                          [data, data, data],
                                          population="RingA",
                                          node_ids=[0, 1, 2])

    with open(create_tmp_simulation_config_file, encoding="utf-8") as fd:
        sim_config_data = json.load(fd)

    sim_config_data["inputs"] = {
        "ex_input_replay": {
            "input_type": "current_clamp",
            "module": "replay",
            "delay": 0.0,
            "duration": 50.0,
            "path": str(path),
            "node_set": "RingA"  #XXX should do all of them?
            }
        }

    with open(create_tmp_simulation_config_file, "w", encoding="utf-8") as fd:
        json.dump(sim_config_data, fd)

    subprocess.run(
        [
            "neurodamus",
            create_tmp_simulation_config_file,
            f"--output-path={tmp_path}/current-replay",
            ],
        check=True,
        capture_output=True,
    )
    breakpoint() # XXX BREAKPOINT
    # compare reports
