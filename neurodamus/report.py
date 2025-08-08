import logging
from enum import IntEnum

from .core import NeuronWrapper as Nd


class Report:
    """Abstract base class for handling simulation reports in NEURON.

    Provides methods for parsing report parameters, handling variables, and defining the structure
    required by subclasses to append specific data (e.g., compartments or currents).
    """

    CURRENT_INJECTING_PROCESSES = {"SEClamp", "IClamp"}

    class ScalingMode(IntEnum):
        """Enum to define scaling modes used in report generation.

        SCALING_NONE: No scaling.
        SCALING_AREA: Scale by membrane area.
        SCALING_ELECTRODE: Custom/electrode-based scaling (fallback/default).
        """

        SCALING_NONE = 0
        SCALING_AREA = 1
        SCALING_ELECTRODE = 2

        @classmethod
        def from_option(cls, option):
            """Map user-provided scaling option to a ScalingMode enum.

            :param option: User-specified string or None.
            :return: Corresponding ScalingMode.
            """
            mapping = {
                None: cls.SCALING_AREA,
                "Area": cls.SCALING_AREA,
                "None": cls.SCALING_NONE,
            }
            return mapping.get(option, cls.SCALING_ELECTRODE)

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

        self.type = params.rep_type.lower()
        self.variables = self.parse_variable_names(params.report_on)

        self.report_dt = params.dt
        self.scaling_mode = self.ScalingMode.from_option(params.scaling)
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
        self, cell_obj, point, vgid, pop_name, pop_offset, sum_currents_into_soma
    ):
        """Abstract method to be implemented by subclasses to add section-level report data.

        :raises NotImplementedError: Always, unless overridden in subclass.
        """
        raise NotImplementedError("Subclasses must implement register_gid_section()")

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
    def get_point_processes(section, mechanism):
        """Retrieve all synapse objects attached to a given section.

        :param section: The NEURON section object to search for synapses.
        :param mechanism: The mechanism requested
        :return: A list of synapse objects attached to the section.
        """
        synapses = [
            syn
            for seg in section
            for syn in seg.point_processes()
            if syn.hname().startswith(mechanism)
        ]
        return synapses

    @staticmethod
    def parse_variable_names(report_on):
        """Parse variable names from a user-specified string into mechanism-variable tuples.

        Also, activate fast_i_mem if necessary.

        E.g., "hh.ina pas.i" → [("hh", "ina"), ("pas", "i")]

        :return: List of (mechanism, variable) tuples.
        """
        tokens_with_vars = []
        tokens = report_on.split()  # Splitting by whitespace

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

        if mechanism != "i_membrane_" and self.scaling_mode == Report.ScalingMode.SCALING_AREA:
            return section(x).area() / 100.0

        return 1.0


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
        self, cell_obj, point, vgid, pop_name, pop_offset, _sum_currents_into_soma
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
            if len(var_refs) != 1:
                raise AttributeError(
                    f"Expected exactly one reference for variable '{variable_name}' "
                    f"of mechanism '{mechanism}' at location {x}, but found {len(var_refs)}."
                )
            section_id = cell_obj.get_section_id(section)
            self.report.AddVar(var_refs[0], section_id, gid, pop_name)


class SummationReport(Report):
    """Concrete Report subclass for summing currents or other variables across sections.

    Handles intrinsic currents and point processes, possibly summing them into soma.
    """

    def register_gid_section(
        self, cell_obj, point, vgid, pop_name, pop_offset, sum_currents_into_soma
    ):
        """Append summed variable data for a given cell across sections.

        :param cell_obj: The cell being reported.
        :param point: Point containing section list and x positions.
        :param vgid: Optional virtual GID.
        :param pop_name: Population name.
        :param pop_offset: Population GID offset.
        :param sum_currents_into_soma: If True, collapses sum into soma.
        """
        if self.use_coreneuron:
            return
        gid = cell_obj.gid
        vgid = vgid or gid

        self.report.AddNode(gid, pop_name, pop_offset)

        if sum_currents_into_soma:
            alu_helper = self.setup_alu_for_summation(0.5)

        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            if not sum_currents_into_soma:
                alu_helper = self.setup_alu_for_summation(x)

            self.process_mechanisms(section, x, alu_helper)

            if not sum_currents_into_soma:
                section_index = cell_obj.get_section_id(section)
                self.add_summation_var_and_commit_alu(alu_helper, section_index, gid, pop_name)
        if sum_currents_into_soma:
            # soma
            self.add_summation_var_and_commit_alu(alu_helper, 0, gid, pop_name)

    def process_mechanisms(self, section, x, alu_helper):
        """Add the ref variable identified by x, mechanism, and variable_name
        to alu_helper multiplied by the scaling_factor.

        Note: compartments without the variable are silently skipped.
        """
        for mechanism, variable in self.variables:
            var_refs = Report.get_var_refs(section, x, mechanism, variable)
            for var_ref in var_refs:
                scaling_factor = self.get_scaling_factor(section, x, mechanism)
                if scaling_factor == 0:
                    logging.warning(
                        "Skipping intrinsic current '%s' at a location with area = 0", mechanism
                    )
                    continue
                alu_helper.addvar(var_ref, scaling_factor)
            # simply skip if not valid. No logging or error is necessary

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
        self, cell_obj, point, vgid, pop_name, pop_offset, _sum_currents_into_soma
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
            raise AttributeError(f"Mechanism '{mechanism}' not found.")
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


NOT_SUPPORTED = object()
_report_classes = {
    "compartment": CompartmentReport,
    "compartment_set": CompartmentReport,
    "summation": SummationReport,
    "synapse": SynapseReport,
    "lfp": NOT_SUPPORTED,
}


def create_report(params, use_coreneuron):
    """Factory function to create a report instance based on parameters."""
    cls = _report_classes.get(params.rep_type.lower())
    if cls is None:
        raise ValueError(f"Unknown report type: {params.rep_type}")
    if cls is NOT_SUPPORTED:
        return None
    return cls(params, use_coreneuron)
