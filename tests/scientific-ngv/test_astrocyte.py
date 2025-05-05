from unittest.mock import Mock

from neurodamus.ngv import Astrocyte

from ..conftest import SIM_DIR

MORPHOLOGY_PATH = SIM_DIR / "ngv" / "sub_circuit" / "morphologies" / "astrocytes"


def test_base():
    # mock circuit_conf and meinfos to pich an astrocyte morphology
    circuit_conf = Mock()
    circuit_conf.MorphologyPath = MORPHOLOGY_PATH
    circuit_conf.MorphologyType = "h5"

    meinfos = Mock()
    meinfos.morph_name = "GLIA_0000000000000"

    # instantiate the astrocyte
    a = Astrocyte(gid=0, meinfos=meinfos, circuit_conf=circuit_conf)

    glut_type = type(a._glut_list[0]) if a._glut_list else None
    soma_type = type(a._soma_glut)
    full_glut_list = list(a.glut_list)

    assert all(isinstance(g, glut_type) for g in a._glut_list), ""
    "All elements in _glut_list must be GlutReceive"
    assert len(full_glut_list) == len(a._glut_list) + 1, ""
    "glut_list should be exactly one longer than _glut_list"
    assert all(isinstance(g, glut_type) for g in full_glut_list[:-1]), ""
    "All but last in glut_list must be GlutReceive"
    assert isinstance(full_glut_list[-1], soma_type), ""
    "Last element in glut_list must be GlutReceiveSoma"
