from quantify.utilities.general import delete_keys_from_dict, make_hash


def test_delete_keys_from_dict():

    test_dict = {"a": 5, "b": 6, "c": {"D": 4, "E": 8}}

    assert "a" in test_dict.keys()
    test_dict = delete_keys_from_dict(test_dict, {"a"})
    assert "a" not in test_dict.keys()

    assert "D" in test_dict['c'].keys()
    test_dict = delete_keys_from_dict(test_dict, {"D"})
    assert "D" not in test_dict['c'].keys()


def test_make_hash():

    my_test_dict = {'a': 5, "nested_dict": {
        "a": 2, "c": 4, "B": 'str'}, 'b': 24}

    same_test_dict_diff_order = {'a': 5, 'b': 24, "nested_dict": {
        "a": 2, "c": 4, "B": 'str'}, }

    diff_test_dict = {"nested_dict": {
        "a": 2, "c": 4, "B": 'str'}, 'b': 24}

    test_hash = make_hash(my_test_dict)
    same_test_hash = make_hash(same_test_dict_diff_order)

    assert test_hash == same_test_hash

    diff_hash = make_hash(diff_test_dict)

    assert test_hash != diff_hash

    # modify dict in place, the object id won't change
    my_test_dict['efg'] = 15
    new_hash = make_hash(my_test_dict)
    assert test_hash != new_hash


