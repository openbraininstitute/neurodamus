from pathlib import Path

import pytest

from neurodamus.morphio_wrapper import MorphIOWrapper, register_legacy_module_alias
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


# A cut-down v6 e-model template: BluePyEModel's `cell_template_neurodamus_sbo.jinja2`, the
# current template per the sonata-extension "HOC E-Model Templates" documentation. Its
# declarations, the start of `init` (up to loading the morphology) and the `load_morphology`,
# `indexSections`, `sec_count` and `geom_nsec` procs are verbatim, except that both quit() calls
# set `load_error` instead, since quit() would terminate the test process. Channel insertion and
# biophysics are left out: they need the e-model's mechanisms and do not touch the morphology.
V6_TEMPLATE = """
begintemplate V6MorphologyCell
  public init, gid, load_error
  public soma, dend, apic, axon, myelin
  create soma[1], dend[1], apic[1], axon[1], myelin[1]
  public nSecAll, nSecSoma, nSecApical, nSecBasal, nSecMyelinated, nSecAxonalOrig, nSecAxonal
  public CellRef
  objref this, CellRef, segCounts

  public all, somatic, apical, axonal, basal, myelinated
  objref all, somatic, apical, axonal, basal, myelinated

proc init(/* args: morphology_dir, morphology_name */) {
  all = new SectionList()
  apical = new SectionList()
  axonal = new SectionList()
  basal = new SectionList()
  somatic = new SectionList()
  myelinated = new SectionList()

  //For compatibility with BBP CCells
  CellRef = this

  forall delete_section()

  gid = $1
  load_error = 0

  load_morphology($s2, $s3)
}

proc indexSections() { local index
    index = 0
    forsec all {
        v(0.0001) = index
        index = index +1
    }
}

proc load_morphology(/* morphology_dir, morphology_name */) {localobj morph, import, sf, extension, commands, pyobj
  strdef morph_path
  sprint(morph_path, "%s/%s", $s1, $s2)
  sf = new StringFunctions()
  extension = new String()
  sscanf(morph_path, "%s", extension.s)

  sf.right(extension.s, sf.len(extension.s)-3)
  if(strcmp(extension.s, "ASC") == 0 ){
    extension.s = "asc"
  }
  if(strcmp(extension.s, "SWC") == 0 ){
    extension.s = "swc"
  }

  if( strcmp(extension.s, "asc") == 0 ) {
     morph = new Import3d_Neurolucida3()
     morph.quiet = 1
     morph.input(morph_path)

     import = new Import3d_GUI(morph, 0)
     import.instantiate(this)
  } else if( strcmp(extension.s, "swc" ) == 0) {
     morph = new Import3d_SWC_read()
     morph.quiet = 1
     morph.input(morph_path)

     import = new Import3d_GUI(morph, 0)
     import.instantiate(this)
  } else if( strcmp(extension.s, ".h5") == 0 ) {
    if(nrnpython ("from morphio_wrapper import MorphIOWrapper") == 1) {
        pyobj = new PythonObject()
        commands = pyobj.MorphIOWrapper(morph_path).morph_as_hoc()
        for i = 0, pyobj.len(commands) - 1 {
            execute(commands._[i], this)
        }
        indexSections()
        geom_nsec()
    } else {
      printf( ".h5 morphlogy used but cannot load 'morphio_wrapper'." )
      load_error = 1
    }
  } else {
    printf(extension.s)
    printf("Unsupported file format: Morphology file has to end with .asc, .swc or .h5" )
    load_error = 2
  }
}

func sec_count(/* SectionList */) { local nSec
  nSec = 0
  forsec $o1 {
      nSec += 1
  }
  return nSec
}

proc geom_nsec() { local nSec
  nSecAll = sec_count(all)
  nSecSoma = sec_count(somatic)
  nSecApical = sec_count(apical)
  nSecBasal = sec_count(basal)
  nSecMyelinated = sec_count(myelinated)
  nSecAxonalOrig = nSecAxonal = sec_count(axonal)

  segCounts = new Vector()
  segCounts.resize(nSecAll)
  nSec = 0
  forsec all {
    segCounts.x[nSec] = nseg
    nSec += 1
  }
}

endtemplate V6MorphologyCell
"""


def test_legacy_module_alias_is_registered(monkeypatch):
    """The reader is exposed as a top-level `morphio_wrapper`, importable from hoc."""
    import sys

    import neurodamus.morphio_wrapper

    monkeypatch.delitem(sys.modules, "morphio_wrapper", raising=False)
    neurodamus.morphio_wrapper.register_legacy_module_alias()

    assert sys.modules["morphio_wrapper"] is neurodamus.morphio_wrapper
    assert Nd.h.nrnpython("from morphio_wrapper import MorphIOWrapper") == 1


def test_v6_template_loads_h5_morphology(tmp_path):
    """The v6 e-model template's h5 branch builds the same cell as MorphIOWrapper."""
    register_legacy_module_alias()
    template = tmp_path / "v6_template.hoc"
    template.write_text(V6_TEMPLATE)
    assert Nd.h.load_file(str(template))

    morph_file = SAMPLE_MORPHOLOGIES / "C060114A7.h5"
    cell = Nd.h.V6MorphologyCell(0, str(morph_file.parent), morph_file.name)
    reference = Nd.Cell(2)
    MorphIOWrapper(morph_file).instantiate(reference)

    assert cell.load_error == 0
    assert _geometry(cell) == _geometry(reference)
    # geom_nsec() ran on the h5 branch
    assert cell.nSecAll == len(list(cell.all))
    assert cell.nSecAxonalOrig == len(list(cell.axonal)) > 0


def test_legacy_module_alias_keeps_existing_module(tmp_path):
    """An installed top-level `morphio_wrapper` is not shadowed by the alias."""
    import os
    import subprocess
    import sys

    (tmp_path / "morphio_wrapper.py").write_text("ORIGIN = 'installed'\n")
    code = (
        "from neurodamus.morphio_wrapper import register_legacy_module_alias\n"
        "register_legacy_module_alias()\n"
        "import morphio_wrapper\n"
        "assert getattr(morphio_wrapper, 'ORIGIN', None) == 'installed', morphio_wrapper\n"
    )
    repo_root = Path(__file__).parents[2]
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(tmp_path), str(repo_root), os.environ.get("PYTHONPATH")) if p
        ),
        "NEURON_INIT_MPI": "0",
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
