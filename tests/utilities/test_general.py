# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import numpy as np

from quantify_core.utilities.general import (
    delete_keys_from_dict,
    make_hash,
    save_json,
    load_json,
)


def test_delete_keys_from_dict() -> None:

    test_dict = {"a": 5, "b": 6, "c": {"D": 4, "E": 8}}

    assert "a" in test_dict.keys()
    test_dict = delete_keys_from_dict(test_dict, {"a"})
    assert "a" not in test_dict.keys()

    assert isinstance(test_dict["c"], dict)
    assert "D" in test_dict["c"].keys()

    test_dict = delete_keys_from_dict(test_dict, {"D"})

    assert isinstance(test_dict["c"], dict)
    assert "D" not in test_dict["c"].keys()


def test_make_hash() -> None:

    my_test_dict = {"a": 5, "nested_dict": {"a": 2, "c": 4, "B": "str"}, "b": 24}

    same_test_dict_diff_order = {
        "a": 5,
        "b": 24,
        "nested_dict": {"a": 2, "c": 4, "B": "str"},
    }

    diff_test_dict = {"nested_dict": {"a": 2, "c": 4, "B": "str"}, "b": 24}

    test_hash = make_hash(my_test_dict)
    same_test_hash = make_hash(same_test_dict_diff_order)

    assert test_hash == same_test_hash

    diff_hash = make_hash(diff_test_dict)

    assert test_hash != diff_hash

    # modify dict in place, the object id won't change
    my_test_dict["efg"] = 15
    new_hash = make_hash(my_test_dict)
    assert test_hash != new_hash


def test_save_load_json(tmp_test_data_dir) -> None:
    data = {"snap": 1, "snap2": "str", "dat": np.array([1.0, 3.14, 5])}
    save_json(directory=tmp_test_data_dir, filename="test_save.json", data=data)
    loaded_data = load_json(full_path=tmp_test_data_dir / "test_save.json")

    assert data.keys() == loaded_data.keys()

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_almost_equal(value, loaded_data.get("dat"))
        else:
            assert value == loaded_data.get(key)
