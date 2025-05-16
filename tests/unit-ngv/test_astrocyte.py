import numpy.testing as npt
from neurodamus.ngv import Astrocyte
from tests.conftest import NGV_DIR
from unittest.mock import MagicMock
from itertools import chain

meinfos = MagicMock()
meinfos.morph_name = "glia"
circuit_conf = MagicMock()
circuit_conf.MorphologyPath = NGV_DIR / "morphologies" / "h5"
circuit_conf.MorphologyType = "h5"


def check_cadifus_pointer(sec, glut):
    """
    Check the pointer by changing the value in GlutReceive
    and checking the change in cadifus
    """
    assert sec(0.5).cadifus.glu2 == glut.glut
    glut.glut += 1
    assert sec(0.5).cadifus.glu2 == glut.glut


def test_init_and_add_endfeet():
    """ Test the basic instantiation of an astrocyte including
    endfeet connections """
    parent_ids = [3, 8, 9]
    lengths = [1.1, 1.2, 1.3]
    diameters = [2.1, 2.2, 2.3]
    R0passes = [3.1, 3.2, 3.3]

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

    # glut_list tests
    glut_list = astro.glut_list
    len(glut_list) == len(astro.glut_all) + len(astro.glut_endfeet) + 1
    glut_receive_type = type(glut_list[0])  # Assuming the first element is GlutReceive
    glut_receive_soma_type = type(glut_list[-1])  # Assuming the last element is GlutReceiveSoma

    # Check that all elements in GlutList are GlutReceive except the last one
    assert all(isinstance(glut, glut_receive_type) for glut in glut_list[:-1])
    assert isinstance(glut_list[-1], glut_receive_soma_type)

    for glut, sec in zip(astro.glut_list, chain(astro.all, astro.endfeet)):
        sec_glut = sec(0.5).point_processes()[-1]
        assert glut.same(sec_glut)
        check_cadifus_pointer(sec, glut)

    assert astro._cellref.soma[0](0.5).point_processes()[0].hname() == "GlutReceiveSoma[0]"
