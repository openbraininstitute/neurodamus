from neurodamus.morphio_wrapper import MorphIOWrapper
from tests.conftest import NGV_DIR


def test_section_names():
    """Check that section names from morph.sections match NEURON
    names after cell instantiation.
    """

    from neurodamus.core import NeuronWrapper as Nd
    morph_file = NGV_DIR / "morphologies" / "h5" / "glia.h5"
    morph = MorphIOWrapper(morph_file)
    cell = Nd.Cell(0)
    cell.AddHocMorph(morph.morph_as_hoc())

    morph_section_names = [f"Cell[0].{i}"
                           for i in morph.section_names]
    nrn_section_names = [i.name() for i in cell.all]

    assert morph_section_names == nrn_section_names
