import pathlib


def get_test_data_dir():
    return pathlib.Path(__name__).parent.resolve() / 'tests' / 'test_data'
