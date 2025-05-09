from neurodamus.ngv import GlutList


def test_glut_list():
    """Test base functionality of glut_list"""

    ll = GlutList(range(5), -1)
    assert ll == [0, 1, 2, 3, 4, -1]
    ll.pop()
    assert ll == [0, 1, 2, 3, -1]
    ll.append(10)
    assert ll == [0, 1, 2, 3, 10, -1]
    ll[2] = 100
    assert ll == [0, 1, 100, 3, 10, -1]
    ll[-1] = -2
    assert ll == [0, 1, 100, 3, 10, -2]
    assert ll.tail == -2
    ll.tail = -3
    assert ll == [0, 1, 100, 3, 10, -3]


# @pytest.mark.parametrize("create_tmp_simulation_config_file", [
#     {
#         "src_dir": str(NGV_DIR),
#         "simconfig_file": "simulation_config.json"
#     }
# ], indirect=True)
# def test_vasccouplingB_radii(create_tmp_simulation_config_file):
#     from neurodamus.core import NeuronWrapper as Nd

#     n = Neurodamus(create_tmp_simulation_config_file)
