from pathlib import Path

from . import NeuronWrapper as Nd
from .configuration import SimConfig


class CompartmentMapping:
    """Interface to register section segment mapping with NEURON."""

    def __init__(self, cell_distributor):
        self.cell_distributor = cell_distributor
        self.pc = Nd.ParallelContext()

    @staticmethod
    def create_section_vectors(section_id, section, secvec, segvec):
        num_segments = 0
        for seg in section:
            secvec.append(section_id)
            segvec.append(seg.node_index())
            num_segments += 1

        return num_segments

    def process_section(self, cell, section, num_electrodes, all_lfp_factors, section_offset):
        secvec, segvec, lfp_factors = Nd.Vector(), Nd.Vector(), Nd.Vector()
        num_segments = 0
        section_attr = getattr(cell._cellref, section[0], None)
        if section_attr:
            for sec in section_attr:
                section_id = cell.get_section_id(sec)
                num_segments += self.create_section_vectors(section_id, sec, secvec, segvec)

        if num_electrodes > 0 and all_lfp_factors.size() > 0 and num_segments > 0:
            start_idx = section_offset * num_electrodes
            end_idx = (section_offset + num_segments) * num_electrodes - 1

            print(start_idx, end_idx, all_lfp_factors)
            

            lfp_factors.copy(all_lfp_factors, start_idx, end_idx)

        self.pc.nrnbbcore_register_mapping(
            cell.gid, section[1], secvec, segvec, lfp_factors, num_electrodes
        )
        return num_segments

    def register_mapping(self):
        # TODO fix+1 this can go
        sections = [
            ("somatic", "soma"),
            ("axonal", "axon"),
            ("basal", "dend"),
            ("apical", "apic"),
            ("AIS", "ais"),
            ("nodal", "node"),
            ("myelinated", "myelin"),
        ]
        gidvec = self.cell_distributor.getGidListForProcessor()
        for activegid in gidvec:
            cell = self.cell_distributor.get_cell(activegid)
            all_lfp_factors = Nd.Vector()
            num_electrodes = 0
            lfp_manager = getattr(self.cell_distributor, "_lfp_manager", None)
            if lfp_manager:
                pop_info = self.cell_distributor.getPopulationInfo(activegid)
                num_electrodes = lfp_manager.get_number_electrodes(activegid, pop_info)
                all_lfp_factors = lfp_manager.read_lfp_factors(activegid, pop_info)

            section_offset = 0
            for section in sections:
                processed_segments = self.process_section(
                    cell, section, num_electrodes, all_lfp_factors, section_offset
                )
                section_offset += processed_segments


class _CoreNEURONConfig:
    """Responsible for managing the configuration of the CoreNEURON simulation.

    It writes the simulation / report configurations and calls the CoreNEURON solver.

    Note: this creates the `CoreConfig` singleton
    """

    default_cell_permute = 0
    artificial_cell_object = None

    @property
    def sim_config_file(self):
        """Get sim config file path to be saved"""
        return str(Path(self.build_path) / "sim.conf")

    @property
    def report_config_file_save(self):
        """Get report config file path to be saved"""
        return str(Path(self.build_path) / "report.conf")

    @property
    def report_config_file_restore(self):
        """Get report config file path to be restored

        We need this file and path for restoring because we cannot recreate it
        from scratch. Only usable when restore exists and is a dir
        """
        return str(Path(SimConfig.restore) / "report.conf")

    @property
    def output_root(self):
        """Get output root from SimConfig"""
        return SimConfig.output_root

    @property
    def datadir(self):
        """Get datadir from SimConfig if not set explicitly"""
        return SimConfig.coreneuron_datadir_path()

    @property
    def build_path(self):
        """Save root folder"""
        return SimConfig.build_path()

    @property
    def restore_path(self):
        """Restore root folder"""
        return SimConfig.restore

    # Instantiates the artificial cell object for CoreNEURON
    # This needs to happen only when CoreNEURON simulation is enabled
    def instantiate_artificial_cell(self):
        self.artificial_cell_object = Nd.CoreNEURONArtificialCell()

    def psolve_core(self, coreneuron_direct_mode=False):
        from neuron import coreneuron

        from . import NeuronWrapper as Nd

        Nd.cvode.cache_efficient(1)
        coreneuron.enable = True
        coreneuron.file_mode = not coreneuron_direct_mode
        coreneuron.sim_config = f"{self.sim_config_file}"
        # set build_path only if the user explicitly asked with --save
        # in this way we do not create 1_2.dat and time.dat if not needed
        if SimConfig.save:
            coreneuron.save_path = self.build_path
        if SimConfig.restore:
            coreneuron.restore_path = self.restore_path

        # Model is already written to disk by calling pc.nrncore_write()
        coreneuron.skip_write_model_to_disk = True
        coreneuron.model_path = f"{self.datadir}"
        Nd.pc.psolve(Nd.tstop)


# Singleton
CoreConfig = _CoreNEURONConfig()
