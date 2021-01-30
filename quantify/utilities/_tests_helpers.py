import pathlib


def get_test_data_dir():
    return pathlib.Path(__file__).parent.parent.parent.resolve() / "tests" / "test_data"
