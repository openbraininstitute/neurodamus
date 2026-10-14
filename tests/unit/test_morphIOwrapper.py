from pathlib import Path

import pytest

from neurodamus.morphio_wrapper import MorphIOWrapper
from tests.conftest import NGV_DIR
from neurodamus.core import NeuronWrapper as Nd


def test_section_names():
    """Check that section names from morph.sections match NEURON
    names after cell instantiation.
    """
    morph_file = NGV_DIR / "morphologies" / "h5" / "glia.h5"
    morph = MorphIOWrapper(morph_file)
    cell = Nd.Cell(0)
    cell.AddHocMorph(morph.morph_as_hoc())

    morph_section_names = [f"Cell[0].{i}"
                           for i in morph._section_names]
    nrn_section_names = [i.name() for i in cell.all]

    assert morph_section_names == nrn_section_names


SAMPLE_MORPHOLOGIES = Path(__file__).parents[1] / "sample_data" / "morphology"


def _geometry(cell):
    """Each section's name, 3D points and parent location, in the order of cell.all"""
    geometry = []
    for sec in cell.all:
        parent = sec.parentseg()
        geometry.append(
            (
                sec.name().split(".", 1)[1],
                [(sec.x3d(i), sec.y3d(i), sec.z3d(i), sec.diam3d(i)) for i in range(sec.n3d())],
                parent and (parent.sec.name().split(".", 1)[1], parent.x),
            )
        )
    return geometry


@pytest.mark.parametrize(
    "morph_file",
    [
        SAMPLE_MORPHOLOGIES / "C060114A7.h5",
        NGV_DIR / "morphologies" / "h5" / "glia.h5",
    ],
)
def test_instantiate_matches_hoc_commands(morph_file):
    """instantiate() builds the same sections, points and topology as the hoc commands."""
    morph = MorphIOWrapper(morph_file)
    from_commands, instantiated = Nd.Cell(0), Nd.Cell(1)
    from_commands.execute_commands(morph.morph_as_hoc())
    morph.instantiate(instantiated)

    assert _geometry(instantiated) == _geometry(from_commands)
