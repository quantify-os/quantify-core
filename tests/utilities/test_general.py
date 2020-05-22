from quantify.utilities.general import delete_keys_from_dict


def test_delete_keys_from_dict():

    test_dict = {"a": 5, "b": 6, "c": {"D": 4, "E": 8}}

    assert "a" in test_dict.keys()
    test_dict = delete_keys_from_dict(test_dict, {"a"})
    assert "a" not in test_dict.keys()

    assert "D" in test_dict['c'].keys()
    test_dict = delete_keys_from_dict(test_dict, {"D"})
    assert "D" not in test_dict['c'].keys()
