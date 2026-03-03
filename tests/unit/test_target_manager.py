import pytest
import libsonata
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

    # List of test cases: (section_type, compartment_type, section_names, expected, check_names)
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

        # filter by section names
        (Sections.all, Compartments.all, {"soma[0]", "dend[1]"},
         [(0, [0, 2, 2], [0.5, 0.25, 0.75]),
          (1, [0, 2, 2], [0.5, 0.25, 0.75]),
          (2, [0, 2, 2], [0.5, 0.25, 0.75])],
         {"soma[0]", "dend[1]"}),

        # filter that produces empty points
        (Sections.soma, Compartments.center, {"dend[0]"},
         [(0, [], []), (1, [], []), (2, [], [])],
         None),
    ]

    for section_type, compartment_type, section_names, expected, check_names in test_cases:
        pts = tgt.get_point_list(
            cell_manager=cell_manager,
            section_type=section_type,
            compartment_type=compartment_type,
            section_names=section_names
        )

        assert len(pts) == len(expected)

        for pt, (exp_gid, exp_ids, exp_xs) in zip(pts, expected, strict=True):
            assert pt.matches(exp_gid, exp_ids, exp_xs)

        if check_names:
            all_names = [sec_ref.sec.name() for pt in pts for sec_ref in pt.sclst]
            if isinstance(check_names, str):
                assert all(check_names in name for name in all_names)
            else:
                assert all(any(req in name for req in check_names) for name in all_names)