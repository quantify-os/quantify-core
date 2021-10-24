# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# this file is added to ensure the tests helpers are discovered by pytest
# see also https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named-yadayadayada

import pytest

from quantify_core.utilities._tests_helpers import (
    get_test_data_dir,
    remove_target_then_copy_from,
    rmdir_recursive,
)


@pytest.fixture(scope="session", autouse=True)
def tmp_test_data_dir(request, tmp_path_factory):
    """
    This is a fixture which uses the pytest tmp_path_factory fixture
    and extends it by copying the entire contents of the test_data
    directory. After the test session is finished, then it calls
    the `cleaup_tmp` method which tears down the fixture and cleans up itself.
    """
    temp_data_dir = tmp_path_factory.mktemp("temp_data")
    remove_target_then_copy_from(source=get_test_data_dir(), target=temp_data_dir)

    def cleanup_tmp():
        rmdir_recursive(root_path=temp_data_dir)

    request.addfinalizer(cleanup_tmp)

    return temp_data_dir
