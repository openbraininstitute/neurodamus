from pathlib import Path

import libsonata
import numpy as np

from . import NeuronWrapper as Nd
from .configuration import SimConfig
from neurodamus.io.lfp_reader import LFPFileReader
from neurodamus.metype import BaseCell


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

    def process_section(
        self, cell, sec_type, sec_list, electrode_offsets, all_lfp_factors, section_offset
    ):
        secvec, segvec, lfp_factors = Nd.Vector(), Nd.Vector(), Nd.Vector()
        num_segments = 0
        for section_id, sec in cell.iter_section_list(sec_list):
            num_segments += self.create_section_vectors(section_id, sec, secvec, segvec)

        num_electrodes = (
            int(electrode_offsets.x[int(electrode_offsets.size()) - 1])
            if electrode_offsets.size() > 0
            else 0
        )
        if num_electrodes > 0 and all_lfp_factors.size() > 0 and num_segments > 0:
            start_idx = section_offset * num_electrodes
            end_idx = (section_offset + num_segments) * num_electrodes - 1
            lfp_factors.copy(all_lfp_factors, start_idx, end_idx)

        self.pc.nrnbbcore_register_mapping(
            cell.gid, sec_type, secvec, segvec, lfp_factors, electrode_offsets
        )
        return num_segments

    @staticmethod
    def _interleave_lfp_factors(readers, gid, pop_info):
        """Build interleaved LFP factors and electrode offsets for a gid.

        Each report provides a (n_compartments, n_electrodes) scaling matrix.
        CoreNEURON expects a flat array where each compartment's electrodes
        from ALL reports are stored contiguously:

            factors_flat = [comp0_repA_e0, comp0_repA_e1, comp0_repB_e0,
                            comp1_repA_e0, comp1_repA_e1, comp1_repB_e0, ...]

        This allows CoreNEURON to read n_total_electrodes values per compartment
        with a simple stride: factors_flat[comp_idx * n_total + electrode_idx].

        Example with 2 compartments, report A (2 electrodes), report B (1 electrode):
            A factors: [[a00, a01], [a10, a11]]
            B factors: [[b00], [b10]]
            Interleaved: [a00, a01, b00, a10, a11, b10]
            electrode_offsets: [0, 2, 3]  (A occupies indices 0-1, B index 2)

        Returns:
            (all_lfp_factors, electrode_offsets) — NEURON Vector and list of ints.
        """
        electrode_offsets = [0]
        matrices = []
        cumulative = 0

        for reader in readers:
            matrix = reader.get_scaling_matrix(gid, pop_info)
            n_elec = matrix.shape[1] if matrix is not None else 0
            cumulative += n_elec
            electrode_offsets.append(cumulative)
            if matrix is not None:
                matrices.append(matrix)

        all_lfp_factors = Nd.Vector()
        if matrices:
            interleaved = np.hstack(matrices).flatten()
            all_lfp_factors.from_python(interleaved.tolist())

        return all_lfp_factors, electrode_offsets

    def register_mapping(self) -> None:
        """Register section-segment and LFP electrode mappings for CoreNEURON.

        For each cell on this rank, registers the section/segment structure with
        NEURON via nrnbbcore_register_mapping. When LFP reports are configured,
        also loads electrode scaling factors from each report's electrodes_file
        and builds a CSR-style electrode_offsets vector across reports.

        A gid not present in a given electrode file contributes zero electrodes
        for that report (the offset increment is zero, factors are empty).

        For example, with report A (3 electrodes, gids 0-2) and report B
        (2 electrodes, gids 2-3):
          - gid 0: offsets=[0,3,3], factors from A only
          - gid 2: offsets=[0,3,5], factors from A+B concatenated
          - gid 3: offsets=[0,0,2], factors from B only
        """
        gidvec = self.cell_distributor.getGidListForProcessor()

        # Open LFP electrode readers from reports (validates structure upfront)
        readers = [
            LFPFileReader(rep_conf.electrodes_file)
            for rep_conf in SimConfig.reports.values()
            if rep_conf.type == libsonata.SimulationConfig.Report.Type.lfp
        ]

        for gid in gidvec:
            cell = self.cell_distributor.get_cell(gid)
            pop_info = self.cell_distributor.getPopulationInfo(gid)
            all_lfp_factors, electrode_offsets = self._interleave_lfp_factors(
                readers, gid, pop_info
            )

            offsets_vec = Nd.Vector(electrode_offsets)

            section_offset = 0
            for sec_type, sec_list in BaseCell.SECTION_TYPES:
                processed_segments = self.process_section(
                    cell,
                    sec_type,
                    sec_list,
                    offsets_vec,
                    all_lfp_factors,
                    section_offset,
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
    def sim_config_file(self) -> str:
        """`sim.conf` path to be saved."""
        return str(Path(self.build_path) / "sim.conf")

    @property
    def report_config_file_save(self) -> str:
        """`report.conf` file path to be saved."""
        return str(Path(self.build_path) / "report.conf")

    @property
    def report_config_file_restore(self) -> str:
        """`report.conf` file path to be restored.

        We need this file and path for restoring because we cannot recreate it
        from scratch. Only usable when restore exists and is a dir
        """
        return str(Path(SimConfig.restore) / "report.conf")

    @property
    def output_root(self):
        """Output root from SimConfig."""
        return SimConfig.output_root

    @property
    def datadir(self):
        """`datadir` from SimConfig if not set explicitly."""
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
