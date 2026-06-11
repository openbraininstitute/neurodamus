import logging

import libsonata
import numpy as np

from .core import NeuronWrapper as Nd
from .report_parameters import ReportParameters
from .target_manager import TargetPointList
from .utils.pyutils import cache_errors


class ReportManager:
    """Registry and factory for Report subclasses based on libsonata report types."""

    _report_types = {}

    @classmethod
    def register_type(cls, report_type):
        """Decorator to register a Report subclass for a given report type."""

        def decorator(report_cls):
            cls._report_types[report_type] = report_cls
            return report_cls

        return decorator

    @classmethod
    @cache_errors
    def create(cls, params: ReportParameters, use_coreneuron, lfp_manager=None):
        """Factory method to create a Report instance for the given parameters.

        Returns:
            Report instance for the given type, or None if the type is
            handled externally (e.g., LFP with CoreNEURON).

        Raises:
            ValueError: If the report type is unknown.
        """
        if params.type == libsonata.SimulationConfig.Report.Type.lfp:
            if use_coreneuron:
                return None  # CoreNEURON handles LFP natively
            return LFPReport(params, use_coreneuron, lfp_manager)  # noqa: F821

        report_cls = cls._report_types.get(params.type)
        if report_cls is None:
            raise ValueError(f"Unknown report type: {params.type.name}")

        return report_cls(params, use_coreneuron)


class Report:
    """Abstract base class for handling simulation reports in NEURON.

    Provides methods for parsing report parameters, handling variables, and defining the structure
    required by subclasses to append specific data (e.g., compartments or currents).
    """

    CURRENT_INJECTING_PROCESSES = {"SEClamp", "IClamp"}

    def __init__(
        self,
        params,
        use_coreneuron,
    ):
        """Initialize a Report object with simulation parameters.

        :param params: Object containing report configuration (e.g., name, dt, unit).
        :param use_coreneuron: Boolean indicating if CoreNEURON is enabled.
        """
        if type(self) is Report:
            raise TypeError("Report is an abstract base class and cannot be instantiated directly.")

        self.type = params.type

        self.variables = self.parse_variable_names(params.report_on)
        self.report_dt = params.dt
        self.scaling = params.scaling
        self.use_coreneuron = use_coreneuron

        self.alu_list = []
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

    def register_gid_section(
        self,
        cell_obj,
        point,
        vgid,
        pop_name,
        pop_offset,
        sections: libsonata.SimulationConfig.Report.Sections,
    ):
        """Abstract method to be implemented by subclasses to add section-level report data.

        :raises NotImplementedError: Always, unless overridden in subclass.
        """
        raise NotImplementedError("Subclasses must implement register_gid_section()")

    @cache_errors
    def setup(self, rep_params: ReportParameters, global_manager):
        for point in rep_params.points:
            gid = point.gid
            pop_name, pop_offset = global_manager.getPopulationInfo(gid)
            cell = global_manager.get_cell(gid)
            spgid = global_manager.getSpGid(gid)

            self.register_gid_section(cell, point, spgid, pop_name, pop_offset, rep_params.sections)

    @staticmethod
    def is_point_process_at_location(point_process, section, x):
        """Check if a point process is located at a specific position within a section.

        :param point_process: The point process to check.
        :param section: The NEURON section in which the point process is located.
        :param x: The normalized position (0 to 1) within the section to check against.
        :return: True if the point process is at the specified position, False otherwise.
        """
        # Get the location of the point process within the section
        # warning: this pushes sec into neuron stack. Remember to pop_section!
        dist = point_process.get_loc()
        # Pop immediately after get_loc to avoid requiring caller to remember
        Nd.pop_section()
        # Calculate the compartment ID based on the location and number of segments
        compartment_id = int(dist * section.nseg)
        # Check if the compartment ID matches the desired location
        return compartment_id == int(x * section.nseg)

    @staticmethod
    def get_point_processes(section, mechanism=None):
        """Retrieve all synapse objects attached to a given section.

        :param section: The NEURON section object to search for synapses.
        :param mechanism: The mechanism requested
        :return: A list of synapse objects attached to the section.
        """
        if mechanism is None:
            return [syn for seg in section for syn in seg.point_processes()]

        return [
            syn
            for seg in section
            for syn in seg.point_processes()
            if syn.hname().startswith(mechanism)
        ]

    @staticmethod
    def parse_variable_names(report_on):
        """Parse variable names from a user-specified string into mechanism-variable tuples.

        Also, activate fast_i_mem if necessary.

        E.g., "hh.ina pas.i" → [("hh", "ina"), ("pas", "i")]

        :return: List of (mechanism, variable) tuples.
        """
        tokens_with_vars = []
        tokens = (t.strip() for t in report_on.split(","))  # Splitting by ,

        for val in tokens:
            if "." in val:
                mechanism, var = val.split(".", 1)  # Splitting by the first period
                tokens_with_vars.append((mechanism, var))
            else:
                if val == "i_membrane":
                    Nd.cvode.use_fast_imem(1)
                    val = "i_membrane_"
                tokens_with_vars.append((val, "i"))  # Default internal variable

        return tokens_with_vars

    @staticmethod
    def get_var_refs(section, x, mechanism, variable_name):
        """Retrieve references to a variable within a mechanism at a specific location on a section.

        This method returns a list of variable references (_ref_<variable_name>) either from:
        - The point processes of the given mechanism located at position x on the section, or
        - Directly from the section or its inserted mechanism if no point processes are present.

        Parameters:
        section : h.Section
            The NEURON section to query.
        x : float
            The location along the section (0 <= x <= 1).
        mechanism : str
            The name of the mechanism or point process.
        variable_name : str
            The name of the variable whose reference is requested.

        Returns:
        list
            A list of variable references (typically hoc references) matching the query.
        """
        point_processes = Report.get_point_processes(section, mechanism)
        # if not a point process, it is a current of voltage. Directly return the reference

        if not point_processes:
            sec_x = section(x)
            var_name = "_ref_" + mechanism
            if hasattr(sec_x, var_name):
                return [getattr(sec_x, var_name)]
            if hasattr(sec_x, mechanism):
                mech = getattr(sec_x, mechanism)
                var_name = "_ref_" + variable_name
                if hasattr(mech, var_name):
                    return [getattr(mech, var_name)]
            return []
        # search among the point processes the ones that at at position x and return the reference
        return [
            getattr(pp, "_ref_" + variable_name)
            for pp in point_processes
            if Report.is_point_process_at_location(pp, section, x)
            and hasattr(pp, "_ref_" + variable_name)
        ]

    def get_scaling_factor(self, section, x, mechanism):
        """Scaling factors for some special variables"""
        if mechanism in self.CURRENT_INJECTING_PROCESSES:
            return -1.0  # Negative for current injecting processes

        if (
            mechanism != "i_membrane_"
            and self.scaling == libsonata.SimulationConfig.Report.Scaling.area
        ):
            return section(x).area() / 100.0

        return 1.0


@ReportManager.register_type(libsonata.SimulationConfig.Report.Type.compartment)
@ReportManager.register_type(libsonata.SimulationConfig.Report.Type.compartment_set)
class CompartmentReport(Report):
    """Concrete Report subclass for reporting compartment-level variables.

    Appends variable references at specific compartment locations for a given cell.
    """

    def __init__(
        self,
        params,
        use_coreneuron,
    ):
        super().__init__(params=params, use_coreneuron=use_coreneuron)
        if len(self.variables) != 1:
            raise ValueError(
                f"Compartment reports requires exactly one variable, "
                f"but received: `{self.variables}`"
            )

    def register_gid_section(
        self,
        cell_obj,
        point,
        vgid,
        pop_name,
        pop_offset,
        _sections: libsonata.SimulationConfig.Report.Sections,
    ):
        """Append section-based report data for a single cell and its compartments.

        :param cell_obj: The cell being processed.
        :param point: Point data containing section list and location.
        :param vgid: Virtual GID to use in report.
        :param pop_name: Population name.
        :param pop_offset: Offset for population indexing.
        :param _sum_currents_into_soma: Unused parameter in this subclass.
        """
        if self.use_coreneuron:
            return
        gid = cell_obj.gid
        vgid = vgid or gid

        mechanism, variable_name = self.variables[0]
        self.report.AddNode(gid, pop_name, pop_offset)
        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            var_refs = self.get_var_refs(section, x, mechanism, variable_name)
            if len(var_refs) == 0:
                raise AttributeError(
                    f"No reference found for variable '{variable_name}' of mechanism '{mechanism}' "
                    f"at location {x}."
                )

            if len(var_refs) > 1:
                raise AttributeError(
                    f"Expected one reference for variable '{variable_name}' "
                    f"of mechanism '{mechanism}' at location "
                    f"{x}, but found {len(var_refs)}. "
                    "Probably many synapses attached to the soma. "
                    "Compartment reports require only one variable per segment."
                )
            section_id = cell_obj.get_section_id(section)
            self.report.AddVar(var_refs[0], section_id, gid, pop_name)


@ReportManager.register_type(libsonata.SimulationConfig.Report.Type.summation)
class SummationReport(Report):
    """Concrete Report subclass for summing currents or other variables across sections.

    Handles intrinsic currents and point processes, possibly summing them into soma.
    """

    def register_gid_section(
        self,
        cell_obj,
        point,
        vgid,
        pop_name,
        pop_offset,
        sections: libsonata.SimulationConfig.Report.Sections,
    ):
        """Append summed variable data for a given cell across sections.

        :param cell_obj: The cell being reported.
        :param point: Point containing section list and x positions.
        :param vgid: Optional virtual GID.
        :param pop_name: Population name.
        :param pop_offset: Population GID offset.
        :param sections: Sum into soma if section is soma

        Note: sections == libsonata.SimulationConfig.Report.Sections.soma
        effectively means that we need
        to sum the values into the soma
        """
        if self.use_coreneuron:
            return
        gid = cell_obj.gid
        vgid = vgid or gid

        self.report.AddNode(gid, pop_name, pop_offset)

        if sections == libsonata.SimulationConfig.Report.Sections.soma:
            alu_helper = self.setup_alu_for_summation(0.5)

        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            if sections == libsonata.SimulationConfig.Report.Sections.all:
                alu_helper = self.setup_alu_for_summation(x)

            self.process_mechanisms(section, x, alu_helper)

            if sections == libsonata.SimulationConfig.Report.Sections.all:
                section_index = cell_obj.get_section_id(section)
                self.add_summation_var_and_commit_alu(alu_helper, section_index, gid, pop_name)
        if sections == libsonata.SimulationConfig.Report.Sections.soma:
            # soma
            self.add_summation_var_and_commit_alu(alu_helper, 0, gid, pop_name)

    def process_mechanisms(self, section, x, alu_helper):
        """Add the ref variable identified by x, mechanism, and variable_name
        to alu_helper multiplied by the scaling_factor.

        Note: compartments without the variable are silently skipped.
        """
        for mechanism, variable in self.variables:
            scaling_factor = self.get_scaling_factor(section, x, mechanism)
            if scaling_factor == 0:
                logging.warning(
                    "Skipping intrinsic current '%s' at a location with area = 0", mechanism
                )
                # simply skip if not valid. No logging or error is necessary
                continue
            var_refs = Report.get_var_refs(section, x, mechanism, variable)
            for var_ref in var_refs:
                alu_helper.addvar(var_ref, scaling_factor)

    def setup_alu_for_summation(self, alu_x):
        """Setup ALU helper for summation."""
        alu_helper = Nd.ALU(alu_x, self.report_dt)
        alu_helper.setop("summation")
        bbss = Nd.BBSaveState()
        bbss.ignore(alu_helper)
        return alu_helper

    def add_summation_var_and_commit_alu(self, alu_helper, section_index, gid, population_name):
        """Add the ALU's output as a summation variable and commit it to the report."""
        self.report.AddVar(alu_helper._ref_output, section_index, gid, population_name)
        # Append ALUhelper to the list of ALU objects
        self.alu_list.append(alu_helper)


@ReportManager.register_type(libsonata.SimulationConfig.Report.Type.synapse)
class SynapseReport(Report):
    def __init__(
        self,
        params,
        use_coreneuron,
    ):
        super().__init__(params=params, use_coreneuron=use_coreneuron)
        if len(self.variables) != 1:
            raise ValueError(
                f"Synapse reports requires exactly one variable, but received: `{self.variables}`"
            )

    def register_gid_section(
        self,
        cell_obj,
        point,
        vgid,
        pop_name,
        pop_offset,
        sections: libsonata.SimulationConfig.Report.Sections,
    ):
        """Append synapse variables for a given cell to the report grouped by gid."""
        gid = cell_obj.gid
        # Default to cell's gid if vgid is not provided
        vgid = vgid or cell_obj.gid

        # Initialize lists for storing synapses and their locations
        synapse_list = []
        mechanism, variable = self.variables[0]
        # Evaluate which synapses to report on
        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            # Iterate over point processes in the section

            point_processes = self.get_point_processes(section, mechanism)
            for synapse in point_processes:
                if self.is_point_process_at_location(synapse, section, x):
                    synapse_list.append(synapse)
                    # Mark synapse as selected for report
                    if hasattr(synapse, "selected_for_report"):
                        synapse.selected_for_report = True

        if not synapse_list:
            raise AttributeError(f"Synapse '{mechanism}' not found on any points on gid: {gid}. ")
        if not self.use_coreneuron:
            # Prepare the report for the cell
            self.report.AddNode(gid, pop_name, pop_offset)
            try:
                for synapse in synapse_list:
                    var_ref = getattr(synapse, "_ref_" + variable)
                    self.report.AddVar(var_ref, synapse.synapseID, gid, pop_name)
            except AttributeError as e:
                msg = f"Variable '{variable}' not found at '{synapse.hname()}'."
                raise AttributeError(msg) from e


class WeightedSummationReport(Report):
    """Report that computes weighted summation of NEURON variables using numpy.

    Mathematical model:
        result[e] = Sum_compartments beta[e, c] * (Sum_variables var_i[c] * alpha_i)

    For LFP: single variable (i_membrane), alpha=1, beta = electrode scaling factors.

    Uses cvode.event at report_dt - 1e-5 to compute values before
    SonataReportHelper samples at report_dt.
    """

    def __init__(self, params, use_coreneuron):
        """Initialize the weighted summation report.

        Args:
            params: ReportParameters with name, dt, start, end, output_dir, unit, report_on.
            use_coreneuron: Boolean indicating if CoreNEURON is enabled.
        """
        super().__init__(params=params, use_coreneuron=use_coreneuron)
        self.start_time = params.start
        self.end_time = params.end

        # Per-GID parallel arrays, populated during register_gid_section()
        self._compartments = []  # list of lists of NEURON segment refs (one per compartment)
        self._betas = []  # list of numpy arrays (n_outputs, n_compartments) or None
        self._output_vecs = []  # list of NEURON Vectors (keeps them alive for SonataReport)
        self._output_nps = []  # cached numpy views into output_vecs (avoids repeated as_numpy())

    @cache_errors
    def setup(self, rep_params, global_manager):
        """Set up the report: collect compartments, load beta weights, register outputs."""
        super().setup(rep_params, global_manager)

        # Start the event chain
        self._schedule_event()

    def register_gid_section(
        self,
        cell_obj,
        point,
        vgid,
        pop_name,
        pop_offset,
        sections: libsonata.SimulationConfig.Report.Sections,
    ):
        """Collect compartments and load beta weights for a single GID."""
        gid = cell_obj.gid

        compartments = self._collect_compartments(point)
        n_compartments = len(compartments)
        if n_compartments == 0:
            logging.warning("GID %d has no compartments for report, skipping.", gid)
            return

        beta_matrix = self._build_beta_matrix(gid, pop_name, pop_offset, n_compartments)
        n_outputs = beta_matrix.shape[0] if beta_matrix is not None else 1

        # Output buffer: NEURON Vector with one element per output channel
        output_vec = Nd.Vector(n_outputs)
        output_np = output_vec.as_numpy()

        # Register with SonataReport
        self.report.AddNode(gid, pop_name, pop_offset)
        for out_idx in range(n_outputs):
            self.report.AddVar(output_vec._ref_x[out_idx], out_idx, gid, pop_name)

        # Store per-GID data in parallel arrays
        self._compartments.append(compartments)
        self._betas.append(beta_matrix)
        self._output_vecs.append(output_vec)
        self._output_nps.append(output_np)

    @staticmethod
    def _collect_compartments(point: TargetPointList) -> list:
        """Collect compartment references (NEURON segments) from a point's section list."""
        compartments = []
        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            compartments.append(section(x))
        return compartments

    def _build_beta_matrix(
        self, gid: int, pop_name: str, pop_offset: int, n_compartments: int
    ) -> np.ndarray | None:
        """Build the beta matrix for a GID.

        Returns:
            np.ndarray (n_outputs, n_compartments): weighted output.
            None: no beta provided, use plain summation.

        Raises:
            ValueError: if beta shape doesn't match compartment count.
        """
        beta_raw = self._get_beta_for_gid(gid, pop_name, pop_offset, n_compartments)
        if beta_raw is not None:
            beta_arr = np.asarray(beta_raw, dtype=np.float64)
            if beta_arr.shape[0] != n_compartments:
                raise ValueError(
                    f"GID {gid}: beta matrix rows ({beta_arr.shape[0]}) "
                    f"!= compartments ({n_compartments})."
                )
            return beta_arr.T  # (n_outputs, n_compartments)
        # No beta provided: plain summation
        return None

    def _get_beta_for_gid(  # noqa: PLR6301
        self, gid: int, pop_name: str, pop_offset: int, n_compartments: int
    ) -> np.ndarray | None:
        """Get beta weights for a GID. Override in subclasses for custom sources.

        Subclasses should return a numpy array of shape (n_compartments, n_outputs)
        or None. None means beta=1 (plain summation to a single output value).
        """
        return None

    def _schedule_event(self):
        """Schedule the next compute event, respecting the reporting window.

        The next event fires at max(start_time, Nd.t + report_dt) - 1e-5,
        ensuring we skip ahead to start_time on the first call and stay
        report_dt-aligned afterwards. The -1e-5 guarantees we compute before
        SonataReportHelper samples.
        """
        event_t = max(self.start_time, Nd.t + self.report_dt) - 1e-5
        if event_t <= self.end_time:
            Nd.cvode.event(event_t, self._compute_callback)

    def restart_event(self):
        """Restart the event chain after restore."""
        self._schedule_event()

    def _compute_callback(self):
        """Self-rescheduling callback that computes the weighted summation.

        Fires at report_dt - 1e-5 so output is ready before SonataReportHelper
        samples at report_dt.
        """
        self._compute()
        self._schedule_event()

    def _compute(self):
        """Compute weighted summation for all report GIDs on this rank."""
        for idx, (beta, output_np) in enumerate(zip(self._betas, self._output_nps, strict=True)):
            values = self._read_inputs(idx)

            # Apply beta weights or plain sum (out= avoids temporary allocation)
            if beta is not None:
                np.dot(beta, values, out=output_np)
            else:
                output_np[0] = values.sum()

    def _read_inputs(self, gid_idx):
        """Read input values from compartments and return as numpy array.

        Override in subclasses to provide variable-specific reading logic.
        """
        raise NotImplementedError("Subclasses must implement _read_inputs")


class LFPReport(WeightedSummationReport):
    """LFP report: weighted summation with electrode scaling factors from HDF5.

    Specializes WeightedSummationReport by loading beta (electrode factors)
    from LFPManager for each GID.
    """

    def __init__(self, params: ReportParameters, use_coreneuron, lfp_manager):
        """Initialize the LFP report.

        Args:
            params: ReportParameters.
            use_coreneuron: Boolean indicating if CoreNEURON is enabled.
            lfp_manager: LFPManager with loaded electrode weights file.
        """
        super().__init__(params=params, use_coreneuron=use_coreneuron)
        self._lfp_manager = lfp_manager

    def _get_beta_for_gid(
        self, gid: int, pop_name: str, pop_offset: int, n_compartments: int
    ) -> np.ndarray | None:
        """Load electrode scaling factors from LFPManager for this GID."""
        pop_info = (pop_name, pop_offset)
        try:
            _, node_id = self._lfp_manager.get_sonata_node_id(gid, pop_info)
            return self._lfp_manager.get_node_id_subsets(node_id, pop_info[0])
        except (KeyError, IndexError) as e:
            logging.warning("GID %d: could not load LFP factors: %s", gid, e)
            return None

    def _read_inputs(self, gid_idx):
        """Read i_membrane_ from each compartment and return as numpy array."""
        compartments = self._compartments[gid_idx]
        return np.array([comp.i_membrane_ for comp in compartments])
