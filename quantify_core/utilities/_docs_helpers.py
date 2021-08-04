# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""Helpers for building docs"""
from __future__ import annotations

from typing import Union
import json
import shutil
import tempfile
from pathlib import Path
import quantify_core.data.handling as dh
from quantify_core.utilities._tests_helpers import get_test_data_dir
from quantify_core.data.types import TUID


def create_tmp_dir_from_test_dataset(tuid: Union[TUID, str]):
    """
    Creates a temporary directory and copies the test dataset into a folder under the
    corresponding date. After using the `tmp_dir` you should call `tmp_dir.cleanup()`.

    Intended to be used in the docs build when access to a dataset is handy.

    NB not intended for doc examples that users are supposed to be able to copy-paste
    and run themselves.

    Parameters
    ----------
    tuid:
        Identifier of the experiment container that will be copied into the
        temporary directory.

    Returns
    -------
    tmp_path:
        A :class:`pathlib.Path` object pointing to the new directory to used as
        :code:`dh.set_datadir(tmp_path)`.
    tmp_dir:
        The :class:`tempfile.TemporaryDirectory` so that :code:`tmp_dir.cleanup()`
        can be called.
    """
    # not calling dh.get_datadir to avoid warning
    old_dir = dh._datadir  # pylint: disable=no-member
    dh.set_datadir(get_test_data_dir())
    exp_container = Path(dh.locate_experiment_container(tuid=tuid))
    if old_dir:
        dh.set_datadir(old_dir)

    tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    tmp_path = Path(tmp_dir.name)
    date_dir_name = exp_container.parent.name

    shutil.copytree(
        exp_container, Path(tmp_dir.name) / date_dir_name / exp_container.name
    )

    return tmp_path, tmp_dir


def notebook_to_rst(notebook_filepath: Path, output_filepath: Path) -> None:
    """
    An experimental utility to covert a Jupyter notebook into an .rst file for writing
    tutorials.

    Intended usage:

    - Use jupytext to pair a .ipynb notebook with a percent script.
    - Version control the .source.py percent script version of the notebook.
    - Generate a `.rst` file with this script to make it part of the docs.
    """

    def get_code_indent(code_cell_source):
        indent = ""
        for line in code_cell_source:
            if line == "\n":
                continue

            for char in line:
                if char == " ":
                    indent += char
                else:
                    break
            break
        return indent

    def make_jupyter_sphinx_block(cell_source):
        indent = get_code_indent(cell_source)
        out = f"\n\n\n{indent}.. jupyter-execute::\n\n"
        indent = f"    {indent}"
        for line in cell_source:
            out += f"{indent}{line}" if line != "\n" else "\n"

        return out

    def make_rst_block(cell_source):
        return "\n\n\n" + "".join(cell_source)

    def cell_to_rst_str(cell):
        cell_type = cell["cell_type"]
        cell_source = cell["source"]

        if cell_type == "code":
            return make_jupyter_sphinx_block(cell_source)

        return make_rst_block(cell_source)

    with open(Path(notebook_filepath), "r") as file:
        json_dict = json.load(file)

    rst_str = ""
    for cell in json_dict["cells"]:
        rst_str += cell_to_rst_str(cell)

    with open(Path(output_filepath), "w") as file:
        file.write(rst_str)
