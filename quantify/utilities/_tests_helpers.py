# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Helpers for testing quantify."""
import pathlib
import os
import shutil


def get_test_data_dir():
    return pathlib.Path(__file__).parent.parent.parent.resolve() / "tests" / "test_data"


def rmdir_recursive(root_path):
    """
    Recursively removes the directory defined by 'root_path' and all the
    contents inside the directory.

    Parameters
    ----------
    root_path : str
        path of the directory to be removed.
    """
    if os.path.exists(root_path):
        shutil.rmtree(root_path)


def remove_target_then_copy_from(source, target):
    """
    Removes the target directory before copying the entire
    content of the source directory to the target.

    Parameters
    ----------
    source : str
        path of the source directory.
    target : str
        path of the target directory
    """
    rmdir_recursive(target)
    shutil.copytree(source, target)
