# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
from textwrap import dedent
import jupytext
from quantify_core.sphinx_extensions.notebook_to_jupyter_sphinx import notebook_to_rst
from sphinx.errors import ExtensionError
import pytest

RST_INDENT = "    "


def strip_rst(rst: str):
    # ignore first two lines, these are just comments
    return "\n".join(rst.split("\n")[2:])


def strp_cmp(str_a: str, str_b: str):
    return str_a.strip() == str_b.strip()


def test_single_cell():
    py_percent_nb = """
    # %%
    1+1"""
    py_percent_nb = dedent(py_percent_nb)

    expected = f"""
    .. jupyter-execute::

    {RST_INDENT}1+1
    """
    expected = dedent(expected)

    rst = notebook_to_rst(
        jupytext.reads(py_percent_nb, fmt="py:percent"), rst_indent=RST_INDENT
    )
    assert strp_cmp(strip_rst(rst), expected)


def test_rst_json_conf():
    py_percent_nb = """
    # %%
    # rst-json-conf: {"indent": "    ", "jupyter_execute_options": [":hide-code:", ":hide-output:"]}
    1+1"""
    py_percent_nb = dedent(py_percent_nb)

    expected = """
    {indent}.. jupyter-execute::
    {indent}    :hide-code:
    {indent}    :hide-output:

    {indent}{indent}1+1
    """
    expected = dedent(expected).format(indent=RST_INDENT)  # format after dedent

    rst = notebook_to_rst(
        jupytext.reads(py_percent_nb, fmt="py:percent"), rst_indent=RST_INDENT
    )
    assert strp_cmp(strip_rst(rst), expected)


def test_rst_conf():
    """Test explicit python dict declaration in the jupyter cell as the conf."""
    py_percent_nb = """
    # %%
    rst_conf = {"indent": "    " * 2, "jupyter_execute_options": [":hide-code:", ":hide-output:"]}
    1+1"""
    py_percent_nb = dedent(py_percent_nb)

    expected = """
    {indent}.. jupyter-execute::
    {indent}    :hide-code:
    {indent}    :hide-output:

    {indent}{r_indent}1+1
    """
    expected = dedent(expected).format(indent=RST_INDENT * 2, r_indent=RST_INDENT)

    rst = notebook_to_rst(
        jupytext.reads(py_percent_nb, fmt="py:percent"), rst_indent=RST_INDENT
    )
    assert strp_cmp(strip_rst(rst), expected)


def test_bad_json():
    py_percent_nb = """
    # %%
    # rst-json-conf: {"indent": False}
    1+1"""
    py_percent_nb = dedent(py_percent_nb)

    with pytest.raises(ExtensionError):
        notebook_to_rst(
            jupytext.reads(py_percent_nb, fmt="py:percent"), rst_indent=RST_INDENT
        )


def test_bad_key():
    py_percent_nb = """
    # %%
    rst_conf = {"bla": "    "}
    1+1"""
    py_percent_nb = dedent(py_percent_nb)

    with pytest.raises(ExtensionError):
        notebook_to_rst(
            jupytext.reads(py_percent_nb, fmt="py:percent"), rst_indent=RST_INDENT
        )
