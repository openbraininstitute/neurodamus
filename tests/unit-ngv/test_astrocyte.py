import numpy.testing as npt
from neurodamus.ngv import Astrocyte
from tests.conftest import NGV_DIR
from neurodamus.metype import METypeItem
from neurodamus.core.configuration import CircuitConfig




def is_cadifus_pointer_set(sec, glut):
    """
    Check the pointer by changing the value in GlutReceive
    and checking the change in cadifus
    """
    if not sec(0.5).cadifus.glu2 == glut.glut:
        return False
    glut.glut += 1
    return sec(0.5).cadifus.glu2 == glut.glut


def test_init_and_add_endfeet():
    """ Test the basic instantiation of an astrocyte including
    endfeet connections """
    parent_ids = [3, 8, 9]
    lengths = [1.1, 1.2, 1.3]
    diameters = [2.1, 2.2, 2.3]
    R0passes = [3.1, 3.2, 3.3]
    meinfos = METypeItem("glia")
    circuit_conf = CircuitConfig(CellLibraryFile="None",
                                 MorphologyPath = NGV_DIR / "morphologies" / "h5",
                                 nrnPath=False,
                                 MorphologyType = "h5")

    astro = Astrocyte(gid=0, meinfos=meinfos, circuit_conf=circuit_conf)
    astro.add_endfeet(parent_ids, lengths, diameters, R0passes)

    assert all(hasattr(sec(0.5), "cadifus") for sec in astro.all)
    assert all(hasattr(sec(0.5), "cadifus") for sec in astro.endfeet)

    # endfeet tests
    assert all(hasattr(sec(0.5), "vascouplingB") for sec in astro.endfeet)
    assert not any(hasattr(sec(0.5), "vascouplingB") for sec in astro.all)

    npt.assert_allclose(lengths, [sec.L for sec in astro.endfeet])
    npt.assert_allclose(diameters, [sec.diam for sec in astro.endfeet])
    npt.assert_allclose(R0passes, [sec(0.5).vascouplingB.R0pas for sec in astro.endfeet])

    glut3 = astro.get_glut(3)
    astro.get_glut(6)
    glut3bis = astro.get_glut(3)
    assert glut3.same(glut3bis)

    # glut_list tests
    glut_list = astro.glut_list
    assert len(glut_list) == 3

    # Check that all elements in GlutList are GlutReceive except the last one
    assert all("GlutReceive" in glut.hname() and
               "GlutReceiveSoma" not in glut.hname() for glut in glut_list[:-1])
    assert "GlutReceiveSoma" in glut_list[-1].hname()

    all_secs = list(astro.all)
    for sec_id, glut in astro._gluts.items():
        sec = all_secs[sec_id]
        assert is_cadifus_pointer_set(sec, glut)

        sec_glut = sec(0.5).point_processes()[-1]
        assert glut.same(sec_glut)
    assert astro._cellref.soma[0](0.5).point_processes()[0].hname() == "GlutReceiveSoma[0]"
