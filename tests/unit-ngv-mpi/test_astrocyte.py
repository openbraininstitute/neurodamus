import pytest

from neurodamus import Neurodamus
from tests.conftest import NGV_DIR


def _check_seg(seg, glut):
    """Verify the GlutReceive mechanism is correctly attached and linked to the segment."""
    assert hasattr(seg, "cadifus")
    assert len(seg.point_processes()) == (2 if "soma" in str(seg) else 1)
    # comparison between hocObjects. -1 because in the soma the
    # first one is GlutReceiveSoma
    assert glut.same(seg.point_processes()[-1])
    # Ensure cadifus.glu2 pointer is synchronized with glut.glut
    assert glut.glut == seg.cadifus.glu2
    v = glut.glut+1
    glut.glut = v
    assert glut.glut == v
    assert glut.glut == seg.cadifus.glu2


@pytest.mark.parametrize("create_tmp_simulation_config_file", [
    {
        "src_dir": str(NGV_DIR),
        "simconfig_file": "simulation_config.json"
    }
], indirect=True)
@pytest.mark.mpi(ranks=1)
def test_astrocyte_point_processes_and_mechanisms(create_tmp_simulation_config_file, mpi_ranks):
    """Check consistency among glut_list, cell point processes, and mechanisms"""
    n = Neurodamus(create_tmp_simulation_config_file)
    astro_manager = n.circuits.get_node_manager("AstrocyteA")
    for idx, cell in enumerate(astro_manager.cells):
        # Check that GlutList length matches section_names + endfeet
        glut_list = list(cell.glut_list)
        section_names = cell.section_names
        endfeet = cell.endfeet or []
        assert len(glut_list) == len(section_names) + len(endfeet) + 1

        # Get the runtime types of GlutReceive and GlutReceiveSoma
        glut_receive_type = type(glut_list[0])  # Assuming the first element is GlutReceive
        glut_receive_soma_type = type(glut_list[-1])  # Assuming the last element is GlutReceiveSoma

        # Check that all elements in GlutList are GlutReceive except the last one
        assert all(isinstance(glut, glut_receive_type) for glut in glut_list[:-1])
        assert isinstance(glut_list[-1], glut_receive_soma_type)

        for sec, glut in zip(cell.CellRef.all, glut_list):
            _check_seg(sec(0.5), glut)

        for sec, glut in zip(endfeet, glut_list[len(cell.CellRef.all):]):
            _check_seg(sec(0.5), glut)
            assert hasattr(sec(0.5), "vascouplingB")

        # check GlutReceiveSoma
        assert cell.CellRef.soma[0](0.5).point_processes()[0].hname() == f"GlutReceiveSoma[{idx}]"
