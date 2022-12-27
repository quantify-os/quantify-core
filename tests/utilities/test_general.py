# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods

import numpy as np
import pytest

from quantify_core.utilities.general import (
    delete_keys_from_dict,
    get_subclasses,
    save_json,
    load_json,
    load_json_safe,
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


def test_get_subclasses() -> None:
    class Foo:
        pass

    class Bar(Foo):
        pass

    class Baz(Bar):
        pass

    class Bing(Baz):
        pass

    classes = set(cls for cls in get_subclasses(Foo))
    assert {Bar, Baz, Bing} == classes, "Only all subclasses"

    classes = set(cls for cls in get_subclasses(Foo, include_base=True))
    assert {Foo, Bar, Baz, Bing} == classes, "Base + all subclasses"

    base = next(get_subclasses(Bing, include_base=True))
    assert base is Bing, "Verify class identity check"

    subclass = next(get_subclasses(Foo))
    assert issubclass(subclass, Foo), "We yield a subclass"
    assert subclass is not Foo, "And that subclass doesn't identify as the base"


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


def test_load_not_existing_json(tmp_test_data_dir) -> None:
    non_existing_file = tmp_test_data_dir / "non_existing_file.json"
    with pytest.raises(FileNotFoundError):
        _ = load_json(full_path=non_existing_file)
    assert load_json_safe(full_path=non_existing_file) is None
