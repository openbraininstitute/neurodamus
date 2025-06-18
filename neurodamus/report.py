import logging
from enum import IntEnum

from .core import NeuronWrapper as Nd


def get_section_index(cell, section):
    """Calculate the global index of a section within a cell, based on section type and local index.

    The function determines the offset for the section type (soma, axon, dend, etc.) and adds
    the section-specific index (e.g., [0], [1], etc.) to compute a unique global index.

    :param cell: The cell instance containing various sections and section counts.
    :param section: The specific NEURON section for which the index is required.
    :return: Integer global index of the section within the cell.
    """
    section_name = str(section)
    base_offset = 0
    section_index = 0
    if "soma" in section_name:
        pass  # base_offset is 0
    elif "axon" in section_name:
        base_offset = cell.nSecSoma
    elif "dend" in section_name:
        base_offset = cell.nSecSoma + cell.nSecAxonalOrig
    elif "apic" in section_name:
        base_offset = cell.nSecSoma + cell.nSecAxonalOrig + cell.nSecBasal
    elif "ais" in section_name:
        base_offset = cell.nSecSoma + cell.nSecAxonalOrig + cell.nSecBasal + cell.nSecApical
    elif "node" in section_name:
        base_offset = (
            cell.nSecSoma
            + cell.nSecAxonalOrig
            + cell.nSecBasal
            + cell.nSecApical
            + getattr(cell, "nSecLastAIS", 0)
        )
    elif "myelin" in section_name:
        base_offset = (
            cell.nSecSoma
            + cell.nSecAxonalOrig
            + cell.nSecBasal
            + cell.nSecApical
            + getattr(cell, "nSecLastAIS", 0)
            + getattr(cell, "nSecNodal", 0)
        )

    # Extract the index from the section name
    index_str = section_name.rsplit("[", maxsplit=1)[-1].rstrip("]")
    section_index = int(index_str)

    return int(base_offset + section_index)


class Report:
    """Abstract base class for handling simulation reports in NEURON.

    Provides methods for parsing report parameters, handling variables, and defining the structure
    required by subclasses to append specific data (e.g., compartments or currents).
    """

    INTRINSIC_CURRENTS = {"i_membrane", "i_membrane_", "ina", "ica", "ik", "i_pas", "i_cap"}
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

        self.variable_name = params.report_on
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

    def append_gid_section(
        self, cell_obj, point, vgid, pop_name, pop_offset, sum_currents_into_soma
    ):
        """Abstract method to be implemented by subclasses to add section-level report data.

        :raises NotImplementedError: Always, unless overridden in subclass.
        """
        raise NotImplementedError("Subclasses must implement append_gid_section()")

    @staticmethod
    def enable_fast_imem(mechanism):
        """Adjust the mechanism name for fast membrane current calculation if necessary.

        If the mechanism is 'i_membrane', enables fast membrane current calculation in NEURON
        and changes the mechanism name to 'i_membrane_'.

        :param mechanism: The original mechanism name.
        :return: The adjusted mechanism name.
        """
        if mechanism == "i_membrane":
            Nd.cvode.use_fast_imem(1)
            mechanism = "i_membrane_"
        return mechanism

    @staticmethod
    def is_point_process_at_location(point_process, section, x):
        """Check if a point process is located at a specific position within a section.

        :param point_process: The point process to check.
        :param section: The NEURON section in which the point process is located.
        :param x: The normalized position (0 to 1) within the section to check against.
        :return: True if the point process is at the specified position, False otherwise.
        """
        # Get the location of the point process within the section
        dist = point_process.get_loc()
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

    def parse_variable_names(self):
        """Parse variable names from a user-specified string into mechanism-variable tuples.

        E.g., "hh.ina pas.i" → [("hh", "ina"), ("pas", "i")]

        :return: List of (mechanism, variable) tuples.
        """
        tokens_with_vars = []
        tokens = self.variable_name.split()  # Splitting by whitespace

        for token in tokens:
            if "." in token:
                mechanism, var = token.split(".", 1)  # Splitting by the first period
                tokens_with_vars.append((mechanism, var))
            else:
                tokens_with_vars.append((token, "i"))  # Default internal variable

        return tokens_with_vars


class CompartmentReport(Report):
    """Concrete Report subclass for reporting compartment-level variables.

    Appends variable references at specific compartment locations for a given cell.
    """

    def append_gid_section(
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

        self.report.AddNode(gid, pop_name, pop_offset)
        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            # Enable fast_imem calculation in Neuron
            self.variable_name = self.enable_fast_imem(self.variable_name)
            var_ref = getattr(section(x), "_ref_" + self.variable_name)
            section_index = get_section_index(cell_obj, section)
            self.report.AddVar(var_ref, section_index, gid, pop_name)


class SummationReport(Report):
    """Concrete Report subclass for summing currents or other variables across sections.

    Handles intrinsic currents and point processes, possibly summing them into soma.
    """

    def append_gid_section(
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
        variable_names = self.parse_variable_names()

        if sum_currents_into_soma:
            alu_helper = self.setup_alu_for_summation(0.5, sum_currents_into_soma)

        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            if not sum_currents_into_soma:
                alu_helper = self.setup_alu_for_summation(x, sum_currents_into_soma)

            self.handle_currents_and_point_processes(section, x, alu_helper, variable_names)

            if not sum_currents_into_soma:
                section_index = get_section_index(cell_obj, section)
                self.add_summation_var_and_commit_alu(alu_helper, section_index, gid, pop_name)
        if sum_currents_into_soma:
            # soma
            self.add_summation_var_and_commit_alu(alu_helper, 0, gid, pop_name)

    def handle_currents_and_point_processes(self, section, x, alu_helper, variable_names):
        """Handle both intrinsic currents and point processes for summation report."""
        area_at_x = section(x).area()
        for mechanism, variable in variable_names:
            self.process_mechanism(section, x, alu_helper, mechanism, variable, area_at_x)

    def process_mechanism(self, section, x, alu_helper, mechanism, variable, area_at_x):
        """Process a single mechanism, whether it's an intrinsic current or a point process."""
        point_processes = self.get_point_processes(section, mechanism)
        if point_processes:
            self.handle_point_processes(section, x, alu_helper, point_processes, variable)
        elif area_at_x:
            self.handle_intrinsic_current(section, x, alu_helper, mechanism, area_at_x)
        else:
            logging.warning(
                "Skipping intrinsic current '%s' at a location with area = 0", mechanism
            )

    def handle_point_processes(self, section, x, alu_helper, point_processes, variable):
        """Handle point processes for a given mechanism."""
        for point_process in point_processes:
            if self.is_point_process_at_location(point_process, section, x):
                is_inverted = any(
                    proc in point_process.hname() for proc in self.CURRENT_INJECTING_PROCESSES
                )
                scalar = -1 if is_inverted else 1
                self.add_variable_to_alu(alu_helper, point_process, variable, scalar)
            Nd.pop_section()

    def handle_intrinsic_current(self, section, x, alu_helper, mechanism, area_at_x):
        """Handle an intrinsic current mechanism."""
        scalar = area_at_x / 100.0 if mechanism != "i_membrane" and self.scaling_mode == 1 else 1
        mechanism = self.enable_fast_imem(mechanism)
        self.add_variable_to_alu(alu_helper, section(x), mechanism, scalar)

    def add_variable_to_alu(self, alu_helper, obj, variable, scalar):
        """Add a variable to the ALU helper with error handling."""
        try:
            var_ref = getattr(obj, "_ref_" + variable)
            alu_helper.addvar(var_ref, scalar)
        except AttributeError:
            if variable in self.INTRINSIC_CURRENTS:
                logging.warning("Current '%s' does not exist at %s", variable, obj)

    def setup_alu_for_summation(self, alu_x, collapsed):
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
    def append_gid_section(
        self, cell_obj, point, vgid, pop_name, pop_offset, _sum_currents_into_soma
    ):
        """Append synapse variables for a given cell to the report grouped by gid."""
        gid = cell_obj.gid
        # Default to cell's gid if vgid is not provided
        vgid = vgid or cell_obj.gid

        # Initialize lists for storing synapses and their locations
        synapse_list = []
        mechanism, variable = self.parse_variable_names()[0]
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
                Nd.pop_section()

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
