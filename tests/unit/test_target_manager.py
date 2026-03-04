import pytest
import libsonata
import numpy as np
Sections = libsonata.SimulationConfig.Report.Sections
Compartments = libsonata.SimulationConfig.Report.Compartments

@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig"}],
    indirect=True,
)
def test_get_point_list_variants(create_tmp_simulation_config_file):
    """Test get_point_list with various configurations
    
    It is just one matrioska test because calling the fixture for multiple tests
    takes some time. I think this setup is still sufficiently clear and is faster
    """
    from neurodamus import Neurodamus

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    tgt = n.target_manager.get_target("RingA")
    cell_manager = n.circuits.get_node_manager("RingA")

    # List of test cases: (section_type, compartment_type, section_local_ids, expected, check_names)
    test_cases = [
        # default (center soma)
        (Sections.soma, Compartments.center, None,
         [(0, [0], [0.5]), (1, [0], [0.5]), (2, [0], [0.5])],
         None),

        # filter by section dend
        (Sections.dend, Compartments.center, None,
         [(0, [1, 2], [0.5, 0.5]), (1, [1, 2], [0.5, 0.5]), (2, [1, 2], [0.5, 0.5])],
         "dend"),

        # no filter (all sections, all compartments)
        (Sections.all, Compartments.all, None,
         [(0, [0, 1, 1, 2, 2], [0.5, 0.25, 0.75, 0.25, 0.75]),
          (1, [0, 1, 1, 2, 2], [0.5, 0.25, 0.75, 0.25, 0.75]),
          (2, [0, 1, 1, 2, 2], [0.5, 0.25, 0.75, 0.25, 0.75])],
         None),

        # filter by section_local_id
        (Sections.dend, Compartments.all, [1],
         [(0, [2, 2], [ 0.25, 0.75]),
          (1, [ 2, 2], [0.25, 0.75]),
          (2, [2, 2], [0.25, 0.75])],
         {"dend[1]"}),

        # skip if section_local_id is not there
        (Sections.dend, Compartments.all, [0, 1, 2],
         [(0, [1, 1, 2, 2], [0.25, 0.75, 0.25, 0.75]),
          (1, [1, 1, 2, 2], [0.25, 0.75, 0.25, 0.75]),
          (2, [1, 1, 2, 2], [0.25, 0.75, 0.25, 0.75])],
         {"dend[0]", "dend[1]"}),
    ]

    for section_type, compartment_type,  section_local_ids, expected, check_names in test_cases:
        pts = tgt.get_point_list(
            cell_manager=cell_manager,
            section_type=section_type,
            compartment_type=compartment_type,
            section_local_ids= section_local_ids
        )

        assert len(pts) == len(expected)

        for pt, (exp_gid, exp_ids, exp_xs) in zip(pts, expected, strict=True):
            assert pt.gid == exp_gid
            assert pt.sclst_ids == exp_ids
            assert np.allclose(pt.x, exp_xs)

        if check_names:
            all_names = [sec_ref.sec.name() for pt in pts for sec_ref in pt.sclst]
            if isinstance(check_names, str):
                assert all(check_names in name for name in all_names)
            else:
                assert all(any(req in name for req in check_names) for name in all_names)


@pytest.mark.parametrize(
    "create_tmp_simulation_config_file",
    [{"simconfig_fixture": "ringtest_baseconfig"}],
    indirect=True,
)
def test_get_point_list_invalid_section_local_ids(create_tmp_simulation_config_file):
    from neurodamus import Neurodamus

    n = Neurodamus(create_tmp_simulation_config_file, disable_reports=True)
    tgt = n.target_manager.get_target("RingA")
    cell_manager = n.circuits.get_node_manager("RingA")

    # 1) Not strictly increasing
    with pytest.raises(AssertionError):
        tgt.get_point_list(
            cell_manager=cell_manager,
            section_type=Sections.dend,
            section_local_ids=[1, 1],
        )

    # 2) Incompatible with Sections.all
    with pytest.raises(AssertionError):
        tgt.get_point_list(
            cell_manager=cell_manager,
            section_type=Sections.all,
            section_local_ids=[0],
        )
