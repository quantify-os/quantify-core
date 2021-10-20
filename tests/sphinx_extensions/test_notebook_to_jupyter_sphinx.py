# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
from textwrap import dedent

import jupytext
import pytest
from sphinx.errors import ExtensionError

from quantify_core.sphinx_extensions.notebook_to_jupyter_sphinx import notebook_to_rst

RST_INDENT = "    "


def strip_rst(rst: str):
    # ignore first two lines, these are just comments
    return "\n".join(rst.split("\n")[2:])


def strp_cmp(rst_from_nb: str, expected_rst: str):
    rst_from_nb = rst_from_nb.strip()
    expected_rst = expected_rst.strip()
    return rst_from_nb == expected_rst


def nb_to_rst(notebook: str):
    notebook = dedent(notebook)
    rst = notebook_to_rst(
        jupytext.reads(notebook, fmt="py:percent"), rst_indent=RST_INDENT
    )
    return strip_rst(rst)


def test_single_cell():
    py_percent_nb = """
    # %%
    1+1"""

    expected = """
    .. jupyter-execute::

        1+1
    """
    expected = dedent(expected)

    assert strp_cmp(nb_to_rst(py_percent_nb), expected)


def test_rst_json_conf():
    py_percent_nb = """
    # %%
    # rst-json-conf: {"indent": "    ", "jupyter_execute_options": [":hide-code:", ":hide-output:"]}
    1+1"""

    expected = """
    {indent}.. jupyter-execute::
    {indent}    :hide-code:
    {indent}    :hide-output:

    {indent}    1+1
    """
    expected = dedent(expected).format(indent=RST_INDENT)  # must format after dedent
    assert strp_cmp(nb_to_rst(py_percent_nb), expected)


def test_rst_conf_space():
    py_percent_nb = """
    # %%
    rst_conf = {"jupyter_execute_options": [":hide-code:"]}

    1+1"""

    expected = """
    .. jupyter-execute::
        :hide-code:

        1+1
    """
    expected = dedent(expected)
    assert strp_cmp(nb_to_rst(py_percent_nb), expected)


def test_rst_conf_comment():
    py_percent_nb = """
    # %%
    rst_conf = {"jupyter_execute_options": [":hide-code:"]}
    # something
    1+1"""

    expected = """
    .. jupyter-execute::
        :hide-code:

        # something
        1+1
    """
    expected = dedent(expected)
    assert strp_cmp(nb_to_rst(py_percent_nb), expected)


def test_rst_conf_only_comments():
    py_percent_nb = """
    # %%
    rst_conf = {"jupyter_execute_options": [":hide-code:"]}
    # something
    # something
    # something
    """

    expected = """
    .. jupyter-execute::
        :hide-code:

        # something
        # something
        # something
    """
    expected = dedent(expected)
    assert strp_cmp(nb_to_rst(py_percent_nb), expected)


def test_rst_conf_only_comments_indent():
    py_percent_nb = """
    # %%
    rst_conf = {"indent": "    "}
    # something
    # something
    """

    expected = """
    {indent}.. jupyter-execute::

    {indent}    # something
    {indent}    # something
    """
    expected = dedent(expected).format(indent=RST_INDENT)
    assert strp_cmp(nb_to_rst(py_percent_nb), expected)


def test_rst_conf_only_comments_indent_space():
    py_percent_nb = """
    # %%
    rst_conf = {"indent": "    "}

    # something
    # something
    """

    expected = """
    {indent}.. jupyter-execute::

    {indent}    # something
    {indent}    # something
    """
    expected = dedent(expected).format(indent=RST_INDENT)
    assert strp_cmp(nb_to_rst(py_percent_nb), expected)


def test_rst_conf():
    """Test explicit python dict declaration in the jupyter cell as the conf."""
    py_percent_nb = """
    # %%
    rst_conf = {"indent": "    " * 2, "jupyter_execute_options": [":hide-code:", ":hide-output:"]}
    1+1"""

    expected = """
    {indent}.. jupyter-execute::
    {indent}    :hide-code:
    {indent}    :hide-output:

    {indent}{r_indent}1+1
    """
    expected = dedent(expected).format(indent=RST_INDENT * 2, r_indent=RST_INDENT)
    assert strp_cmp(nb_to_rst(py_percent_nb), expected)


def test_rst_conf_multi_line():
    """Test that the python dictionary can span multiple lines."""
    py_percent_nb = """
    # %%
    rst_conf = {
        "jupyter_execute_options": [
            ":hide-code:", ":hide-output:", ":code-below:", ":linenos:"
        ]
    }

    1+1"""

    expected = """
    .. jupyter-execute::
        :hide-code:
        :hide-output:
        :code-below:
        :linenos:

        1+1
    """
    expected = dedent(expected)
    assert strp_cmp(nb_to_rst(py_percent_nb), expected)


def test_bad_json():
    py_percent_nb = """
    # %%
    # rst-json-conf: {"indent": False}
    1+1"""

    with pytest.raises(ExtensionError):
        nb_to_rst(py_percent_nb)


def test_bad_key():
    py_percent_nb = """
    # %%
    rst_conf = {"bla": "    "}
    1+1"""

    with pytest.raises(ExtensionError):
        nb_to_rst(py_percent_nb)
