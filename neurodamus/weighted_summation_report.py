"""Numpy-based weighted summation report for NEURON.

Implements a general-purpose report that computes weighted sums of variables
across compartments using numpy vectorized operations. Designed to replace
ALU-based per-compartment summation with bulk array operations.

Initial use case: LFP reporting with NEURON (currently CoreNEURON-only).
Future: replace existing ALU-based summation report.
"""

import logging

import numpy as np

from .core import NeuronWrapper as Nd
from .report import Report


class WeightedSummationReport:
    """Report that computes weighted summation of NEURON variables using numpy.

    Mathematical model:
        result[e] = Sum_compartments beta[e, c] * (Sum_variables var_i[c] * alpha_i)

    For LFP: single variable (i_membrane), alpha=1, beta = electrode scaling factors.

    Uses cvode.event at report_dt - 1e-5 to compute values before
    SonataReportHelper samples at report_dt.
    """

    def __init__(self, params, lfp_manager):
        """Initialize the weighted summation report.

        Args:
            params: ReportParameters with name, dt, start, end, output_dir, unit, report_on.
            lfp_manager: LFPManager instance for loading electrode weights (beta matrix).
        """
        self.report_dt = params.dt
        self.start_time = params.start
        self.end_time = params.end
        self._lfp_manager = lfp_manager

        self.variables = Report.parse_variable_names(params.report_on)

        # Per-GID data: populated during setup()
        # Each entry: {gid, segments, beta, output_vec, output_np}
        self._gid_data = []

        # NEURON SonataReport instance for output
        self.report = Nd.SonataReport(
            0.5,
            params.name,
            params.output_dir,
            params.start,
            params.end,
            params.dt,
            params.unit,
            "compartment",
        )
        Nd.BBSaveState().ignore(self.report)

        # Buffer freshness timestamp for safety checks
        self._buffer_time = -1.0

    def setup(self, rep_params, global_manager):
        """Set up the report: collect segments, load beta weights, register outputs.

        Args:
            rep_params: ReportParameters (with points already set).
            global_manager: GlobalCellManager for accessing cells and population info.
        """
        for point in rep_params.points:
            gid_entry = self._setup_gid(point, global_manager)
            if gid_entry is not None:
                self._gid_data.append(gid_entry)

        # Pre-allocate a shared values buffer sized to the max compartment count
        max_comp = max((d["n_compartments"] for d in self._gid_data), default=0)
        self._values_buffer = np.empty(max_comp, dtype=np.float64)

        # Schedule the first compute event
        self._schedule_event()

    def _setup_gid(self, point, global_manager):
        """Set up report data for a single GID.

        Returns:
            Dict with per-GID data, or None if GID should be skipped.
        """
        gid = point.gid
        pop_name, pop_offset = global_manager.getPopulationInfo(gid)

        # Collect all segment references for this GID
        segments = []
        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            segments.append(section(x))

        n_compartments = len(segments)
        if n_compartments == 0:
            logging.warning("GID %d has no compartments for LFP report, skipping.", gid)
            return None

        # Load beta weights from LFP manager
        pop_info = global_manager.getPopulationInfo(gid)
        _, node_id = self._lfp_manager.get_sonata_node_id(gid, pop_info)
        beta = self._lfp_manager.get_node_id_subsets(node_id, pop_info[0])

        # beta shape from HDF5: (n_compartments, n_electrodes)
        # We need (n_electrodes, n_compartments) for matmul: result = beta.T @ values
        if beta.shape[0] != n_compartments:
            logging.warning(
                "GID %d: beta matrix rows (%d) != compartments (%d). Skipping.",
                gid,
                beta.shape[0],
                n_compartments,
            )
            return None

        beta_matrix = np.asarray(beta, dtype=np.float64).T  # (n_electrodes, n_compartments)
        n_electrodes = beta_matrix.shape[0]

        # Output buffer: NEURON Vector with one element per electrode
        output_vec = Nd.Vector(n_electrodes)
        output_np = output_vec.as_numpy()

        # Register with SonataReport
        self.report.AddNode(gid, pop_name, pop_offset)
        for e_idx in range(n_electrodes):
            self.report.AddVar(output_vec._ref_x[e_idx], e_idx, gid, pop_name)

        return {
            "gid": gid,
            "segments": segments,
            "beta": beta_matrix,
            "output_vec": output_vec,
            "output_np": output_np,
            "n_compartments": n_compartments,
        }

    def _schedule_event(self):
        """Schedule the next compute event via cvode.event."""
        Nd.cvode.event(Nd.t + self.report_dt - 1e-5, self._compute_callback)

    def restart_event(self):
        """Restart the event chain after restore."""
        self._schedule_event()

    def _compute_callback(self):
        """Self-rescheduling callback that computes the weighted summation.

        Fires at report_dt - 1e-5 so output is ready before SonataReportHelper
        samples at report_dt.
        """
        if self.start_time <= Nd.t <= self.end_time:
            self._compute_all_gids()
            self._buffer_time = Nd.t

        # Always reschedule (like ALU fires from t=0 always)
        Nd.cvode.event(Nd.t + self.report_dt - 1e-5, self._compute_callback)

    def _compute_all_gids(self):
        """Compute weighted summation for all report GIDs on this rank."""
        for gid_data in self._gid_data:
            segments = gid_data["segments"]
            beta = gid_data["beta"]
            output_np = gid_data["output_np"]
            n_comp = gid_data["n_compartments"]

            # Read current variable values from segments
            values = self._values_buffer[:n_comp]
            for i, seg in enumerate(segments):
                values[i] = seg.i_membrane_

            # Compute: output = beta @ values
            np.dot(beta, values, out=output_np)

    def _read_values_for_gid(self, gid_data):
        """Read variable values from segments into numpy array.

        Currently reads i_membrane_ only (LFP case).
        Future: support multiple variables with alpha weights.
        """
        segments = gid_data["segments"]
        n_comp = gid_data["n_compartments"]
        values = self._values_buffer[:n_comp]
        for i, seg in enumerate(segments):
            values[i] = seg.i_membrane_
        return values
