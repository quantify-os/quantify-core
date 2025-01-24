# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods

import bz2
import gzip
import json
import lzma
import numpy as np
from pathlib import Path
import pytest

from quantify_core.utilities.general import (
    delete_keys_from_dict,
    get_keys_containing,
    get_subclasses,
    load_json,
    load_json_safe,
    save_json,
    without,
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


def test_without():
    input_dict = {"foo": 1, "bar": 2, "baz": 3}

    actual = without(input_dict, "foo")
    expected = {"bar": 2, "baz": 3}
    assert actual == expected

    actual = without(input_dict, ["foo", "bar"])
    expected = {"baz": 3}
    assert actual == expected

    actual = without(input_dict, ["foo", "bax"])
    expected = {"bar": 2, "baz": 3}
    assert actual == expected


def test_get_keys_containing() -> None:
    test_dict = {"x0": [1, 2, 3], "y0": [4, 5, 6], "x1": [7, 8, 9], "other_key": 79}

    matching_keys = get_keys_containing(test_dict, "x")
    assert matching_keys == {"x0", "x1"}

    matching_zero = get_keys_containing(test_dict, "0")
    assert matching_zero == {"x0", "y0"}

    matching_none = get_keys_containing(test_dict, "z")
    assert matching_none == set()

    matching_empty = get_keys_containing(test_dict, "")
    assert matching_empty == {"x0", "y0", "x1", "other_key"}


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


def test_save_load_json_with_compression(tmp_test_data_dir) -> None:
    data = {"snap": 1, "snap2": "str", "dat": np.array([1.0, 3.14, 5])}

    save_json(
        directory=tmp_test_data_dir,
        filename="test_save_compressed_bz2.json",
        data=data,
        compression="bz2",
    )
    loaded_data = load_json(
        full_path=tmp_test_data_dir / "test_save_compressed_bz2.json"
    )
    assert data.keys() == loaded_data.keys()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_almost_equal(value, loaded_data.get("dat"))
        else:
            assert value == loaded_data.get(key)

    save_json(
        directory=tmp_test_data_dir,
        filename="test_save_compressed_gzip.json",
        data=data,
        compression="gzip",
    )
    loaded_data = load_json(
        full_path=tmp_test_data_dir / "test_save_compressed_gzip.json"
    )
    assert data.keys() == loaded_data.keys()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_almost_equal(value, loaded_data.get("dat"))
        else:
            assert value == loaded_data.get(key)

    save_json(
        directory=tmp_test_data_dir,
        filename="test_save_compressed_lzma.json",
        data=data,
        compression="lzma",
    )
    loaded_data = load_json(
        full_path=tmp_test_data_dir / "test_save_compressed_lzma.json"
    )
    assert data.keys() == loaded_data.keys()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_almost_equal(value, loaded_data.get("dat"))
        else:
            assert value == loaded_data.get(key)


def test_save_json_with_string_directory(tmp_test_data_dir) -> None:
    data = {"test": "data"}
    dir_as_string = str(tmp_test_data_dir)

    save_json(directory=dir_as_string, filename="test_string_dir.json", data=data)

    loaded_data = load_json(tmp_test_data_dir / "test_string_dir.json")
    assert loaded_data == data


def test_save_json_non_existing_directory(tmp_test_data_dir: Path) -> None:
    non_existent_dir = tmp_test_data_dir / "non_existent_dir"
    filename = "test_file.json"
    test_data = {"key": "value"}

    with pytest.raises(FileNotFoundError) as exc_info:
        save_json(non_existent_dir, filename, test_data)

    assert str(exc_info.value) == f"Directory {non_existent_dir} does not exist"


def test_save_json_invalid_filename_type(tmp_test_data_dir: Path) -> None:
    test_dir = tmp_test_data_dir / "test_directory"
    test_dir.mkdir()
    invalid_filename = 123
    test_data = {"key": "value"}

    with pytest.raises(ValueError) as exc_info:
        save_json(test_dir, invalid_filename, test_data)

    assert str(exc_info.value) == "Filename must be a string"


def test_save_json_invalid_compression(tmp_test_data_dir) -> None:
    data = {"test": "data"}
    compression_type = "zip"

    with pytest.raises(ValueError) as exc_info:
        save_json(
            directory=tmp_test_data_dir,
            filename="test.json",
            data=data,
            compression=compression_type,
        )
    assert str(exc_info.value) == (
        f"Unsupported compression type '{compression_type}'. "
        "Supported types are 'bz2', 'gzip', 'lzma'."
    )


def test_load_json(tmp_path) -> None:
    data = {"key": "value"}
    file_path = tmp_path / "test.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    loaded_data = load_json(file_path)
    assert loaded_data == data


def test_load_json_file_not_found(tmp_path) -> None:
    file_path = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        load_json(file_path)


def test_load_json_all_formats_missing(tmp_path):
    file_path = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_json(file_path)
    assert "File not found in any format" in str(exc_info.value)


def test_load_json_compressed_bz2(tmp_path) -> None:
    data = {"key": "value"}
    file_path = tmp_path / "test.json.bz2"
    with bz2.open(file_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    loaded_data = load_json(file_path)
    assert loaded_data == data


def test_load_json_compressed_gzip(tmp_path) -> None:
    data = {"key": "value"}
    file_path = tmp_path / "test.json.gz"
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    loaded_data = load_json(file_path)
    assert loaded_data == data


def test_load_json_compressed_lzma(tmp_path) -> None:
    data = {"key": "value"}
    file_path = tmp_path / "test.json.xz"
    with lzma.open(file_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    loaded_data = load_json(file_path)
    assert loaded_data == data


def test_load_json_compressed_file_not_found(tmp_path) -> None:
    file_path = tmp_path / "non_existent.json.bz2"
    with pytest.raises(FileNotFoundError):
        load_json(file_path)


def test_load_json_corrupted_compressed_file(tmp_path):
    file_path = tmp_path / "corrupted.json.bz2"
    with open(file_path, "wb") as f:
        f.write(b"Not a valid BZ2 file content")

    with pytest.raises(OSError, match="Invalid data stream") as exc_info:
        load_json(file_path)

    assert "Invalid data stream" in str(exc_info.value)


def test_load_json_safe_success(tmp_test_data_dir: Path) -> None:
    data = {"test": "data"}
    test_file = tmp_test_data_dir / "test.json"
    save_json(directory=tmp_test_data_dir, filename="test.json", data=data)
    assert load_json_safe(test_file) == data


def test_load_json_safe_file_not_found(tmp_test_data_dir: Path) -> None:
    non_existing = tmp_test_data_dir / "non_existing.json"
    result = load_json_safe(non_existing)
    assert result is None


def test_load_json_safe_invalid_json(tmp_test_data_dir: Path) -> None:
    invalid_json = tmp_test_data_dir / "invalid.json"
    invalid_json.write_text("not valid json")
    result = load_json_safe(invalid_json)
    assert result is None


def test_load_json_safe_type_error() -> None:
    result = load_json_safe(None)
    assert result is None
