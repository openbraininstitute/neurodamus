import logging

from .core import NeuronWrapper as Nd
from .metype import BaseCell


class Report:
    INTRINSIC_CURRENTS = {"i_membrane", "i_membrane_", "ina", "ica", "ik", "i_pas", "i_cap"}
    CURRENT_INJECTING_PROCESSES = {"SEClamp", "IClamp"}

    def __init__(
        self,
        report_name,
        report_type,
        variable_name,
        unit,
        _format,
        dt,
        start_time,
        end_time,
        output_dir,
        scaling_option=None,
        use_coreneuron=False,
    ):
        self.variable_name = variable_name
        self.report_dt = dt
        self.scaling_mode = self.determine_scaling_mode(scaling_option)
        self.use_coreneuron = use_coreneuron

        self.alu_list = []
        self.report = Nd.SonataReport(
            0.5, report_name, output_dir, start_time, end_time, dt, unit, "compartment"
        )
        Nd.BBSaveState().ignore(self.report)

    @staticmethod
    def determine_scaling_mode(scaling_option):
        if scaling_option is None or scaling_option == "Area":
            return 1  # SCALING_AREA
        if scaling_option == "None":
            return 0  # SCALING_NONE
        return 2  # SCALING_ELECTRODE

    def add_compartment_report(self, cell_obj, point, vgid, pop_name, pop_offset):
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
            section_id = BaseCell.get_section_id(cell_obj, section)
            self.report.AddVar(var_ref, section_id, gid, pop_name)

    def add_summation_report(
        self, cell_obj, point, collapsed, vgid, pop_name="default", pop_offset=0
    ):
        if self.use_coreneuron:
            return
        gid = cell_obj.gid
        vgid = vgid or gid

        self.report.AddNode(gid, pop_name, pop_offset)
        variable_names = self.parse_variable_names()

        if collapsed:
            alu_helper = self.setup_alu_for_summation(0.5, collapsed)

        for i, sc in enumerate(point.sclst):
            section = sc.sec
            x = point.x[i]
            if not collapsed:
                alu_helper = self.setup_alu_for_summation(x, collapsed)

            self.handle_currents_and_point_processes(section, x, alu_helper, variable_names)

            if not collapsed:
                section_id = BaseCell.get_section_id(cell_obj, section)
                self.add_summation_var_and_commit_alu(alu_helper, section_id, gid, pop_name)
        if collapsed:
            # soma
            self.add_summation_var_and_commit_alu(alu_helper, 0, gid, pop_name)

    def add_synapse_report(self, cell_obj, point, vgid, pop_name="default", pop_offset=0):
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
        tokens_with_vars = []
        tokens = self.variable_name.split()  # Splitting by whitespace

        for token in tokens:
            if "." in token:
                mechanism, var = token.split(".", 1)  # Splitting by the first period
                tokens_with_vars.append((mechanism, var))
            else:
                tokens_with_vars.append((token, "i"))  # Default internal variable

        return tokens_with_vars

    def add_summation_var_and_commit_alu(self, alu_helper, section_id, gid, population_name):
        self.report.AddVar(alu_helper._ref_output, section_id, gid, population_name)
        # Append ALUhelper to the list of ALU objects
        self.alu_list.append(alu_helper)
