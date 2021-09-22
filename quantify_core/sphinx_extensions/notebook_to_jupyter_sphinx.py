# Repository: https://gitlab.com/quantify-os/quantify-core
# Licensed according to the LICENCE file on the master branch
"""
A sphinx extension that converts python Jupyter notebook scripts (:code:`.rst.py`) in
the percent format to :code:`.rst` files (to be executed by sphinx).

The extension purpose is to minimize the required overhead for writing and modifying
executable tutorials.

The rationale is to keep things as simple as possible and as easy to debug as possible:

- The code cells are converted into :code:`.. jupyter-execute::` rst directives.
- Raw cells are copy-pasted directly, therefore, they should contain rst contents only.
- Cells in markdown format are ignored.
- The generated :code:`.rst` output files are written to disk for easy inspection. Note that any problems with the rst text will be flagged by sphinx as coming from the output file of this extension. But you are able to insect it to identify the issue (and correct it in the notebook itself!).

Known alternative
-----------------

An alternative to this extensions is to use `nbsphinx in combination with jupytext <https://nbsphinx.readthedocs.io/en/latest/custom-formats.html#Example:-Jupytext>`_\. It has neat features, e.g. correctly pointing to the ``.py`` source file. However, `nbsphinx` has some limitations and potentially complicated-to-install dependencies (like pandoc). Such limitations include:

- It is not possible to insert notebook cells inside ``rst`` directives for example inside a drop-down ``.. note::`` directive.
- Specifying that a raw cell is to be interpreted as ``rst`` is `tricky <https://nbsphinx.readthedocs.io/en/latest/raw-cells.html>`_ and does not seem to be supported in Jupyter Lab.

.. _sec-sphinx-extension-usage:

Usage
-----

1. Create a Jupyter notebook in the `percent format <https://jupytext.readthedocs.io/en/latest/formats.html#the-percent-format>`_ with an extra suffix (:code:`.rst.py`). The extra :code:`.rst` suffix is necessary in order to collect the files that are to be converted.

    - You can also start with a normal notebook :code:`.rst.ipynb` paired with with `.rst.py` percent-formatted script. This is achieved, e.g., with the `jupytext extension <https://jupytext.readthedocs.io/>`_ for Jupyter Lab.
    - Why `percent format`? To keep the scripts compatible with Jupyter and most IDEs.

2. Version control only the :code:`.rst.py` file. Do not commit the :code:`.rst` nor the :code:`.ipynb` files.

3. Add this extension to your sphinx :code:`conf.py` file.

    .. code-block:: python

        extensions = [
            # ...,
            "quantify_core.sphinx_extensions.notebook_to_jupyter_sphinx",
        ]

4. Add the `.rst.py` file(s) in the same location where you would like the `.rst` output file(s) to be generated.

5. Add the file(s) to a table of contents as you would usually do for normal `.rst` file(s). Mind that you do not need to specify the file extension, however, if you do, it must be :code:`.rst` (and not :code:`.rst.py`!).

6. Every time the docs are built by sphinx, the :code:`.rst` file(s) corresponding to all the :code:`.rst.py` file(s) will be generated under the same directory with the same name. This step will be executed right after sphinx loads its settings from the :code:`conf.py` file.

.. note::

    This extension will not process all :code:`.rst.py` files but will only write to
    disk the files that result in different contents compared to the contents of the
    existing :code:`.rst` file. Since sphinx is efficient and does not process files
    that have not changed, this speeds up the development time.

    If you are updating the code that is used in the notebooks you might want to force
    the rebuild of the ``.rst`` files by adding (temporarily) to the ``conf.py``:

    .. code-block::

        # ...
        notebook_to_jupyter_sphinx_always_rebuild = True
        # ...

Code cells configuration magic comment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is necessary to pass some configuration options to this extension in order
for it to produce the indented output from code cells. To achieve this a magic comment
is used, currently supporting two configuration keys. The configuration is a dictionary
that will be parsed as json.

.. note::

    An experienced reader might suggest using the metadata of cells for this task, which is a more "clean" way of storing this information. Nonetheless, it would be more difficult for non-experienced users to understand it and edit the "hidden" metadata of a cell in a notebook environment.

.. code-block:: python

    # rst-json-conf: {"indent": "    ", "jupyter_execute_options": [":hide-output:", ...]}

    # ... the rest of the python code in the cell...

The :code:`"indent"` entry specifies the indentation of the
:code:`.. jupyter-execute::` block produced.
You will need this when you intended the block to be included, e.g., inside a
:code:`.. note::`.
You might argue that you could just indent the code in the cell instead, which works in,
e.g., Jupyter Lab, however the :code:`.rst.py` file will become an invalid python file,
confuse auto formatters and linters, etc..

The :code:`"jupyter_execute_options"` entry is a list of directive options that will be
placed on the line below the :code:`.. jupyter-execute::`.

The above example will produce the following in the :code:`.rst` file :

.. code-block:: rst

    .. jupyter-execute::
        :hide-output:

        # ... the rest of the python code in the cell...

.. tip::

    If you wan to suppress the output of a final line in a notebook cell you could
    usually use a :code:`;`. However, if you use a python auto formatter like black in
    the repository, it will get removed.
    To achieve the same effect assign the output of the last line of a cell to the
    :code:`_` variable. E.g., :code:`_ = plt.plot(...)`. You can read more about this
    python feature
    `here <https://www.datacamp.com/community/tutorials/role-underscore-python>`_.

Possible enhancements
---------------------

The extension could be enhanced in a few ways:

- Include the raw rst cells in the notebooks that `jupyter_sphinx` allows to download.
- Make the "View page source"/"Edit on GitHub/GitLab" point to the ``.rst.py`` script instead of the ``.rst``.
- A Jupyter Lab or browser extension for ``rst`` code highlighting (see limitation below).
- Support for using markdown cells directly with conversion to .rst using a tool like MYST. 

Known limitations
-----------------

Code highlighting in Jupyter Lab
    Unfortunately it seems that it is not possible to make Jupyter Lab highlight the rst code in the (raw) cells of a notebook, which would be useful for this extension. There are some workarounds for Jupyter Notebook involving cell magics but it is not
    quite worth the effort.

Notebook template
-----------------

To make use of this extensions you can start from this ``template.rst.py``.

.. code-block:: python

    # ---
    # jupyter:
    #   jupytext:
    #     cell_markers: '\"\"\"'
    #     formats: py:percent
    #     text_representation:
    #       extension: .py
    #       format_name: percent
    #   kernelspec:
    #     display_name: Python 3 (ipykernel)
    #     language: python
    #     name: python3
    # ---

    # %% [raw]
    \"\"\"
    The contents of this raw cell will be copy-pasted into the ``.rst`` file.
    \"\"\"

    # %%
    # This is a code cell, will be translated into a `.. jupyter-execute::` block.
    assert 1+1 == 2

Place it in the desired location, rename it and navigate to its location using file
browser in Jupyter Lab. Then right-click the file and under the `Open With` select
`Notebook`. Note that you need a relatively recent version of Jupyter Lab for this
to already be part of the Jupyter Lab interface by default (if not consult the
`jupytext documentation <https://jupytext.readthedocs.io>`_).

The ``cell_markers`` in the header of the template tells `jupytext` to store the
contents of raw notebook cells in the ``.rst.py`` files inside blocks that look like
this:

.. code-block:: python

    # %% [raw]
    \"\"\"
    Raw cell contents
    goes here
    \"\"\"

Instead of the default:

.. code-block:: python

    # %% [raw]
    # Raw cell contents
    # goes here

You can remove that line if you wish to use the default representation.

"""  # pylint: disable=line-too-long

from __future__ import annotations

from typing import List, Tuple
import json
from pathlib import Path
import jupytext
from sphinx.errors import ExtensionError
from sphinx.util import logging

logger = logging.getLogger(__name__)

# pylint: disable=unused-argument
def get_code_indent_and_processed_lines(
    cell_source_code: str,
) -> Tuple[str, List[str]]:
    """
    Processes a code cell applying configuration from the magic comment.

    Parameters
    ----------
    cell_source_code
        String containing the code of the cell.
    """
    code_cell_lines = cell_source_code.split("\n")
    indent = ""
    directive_options = []
    if code_cell_lines:
        # skip line in case of a cell magic used for code highlight in Jupyter Notebook
        if code_cell_lines[0] == r"%%rst":
            code_cell_lines = code_cell_lines[1:]

        first_line = code_cell_lines[0]
        magic_comment = "# rst-json-conf:"

        if first_line.startswith(magic_comment):
            conf = json.loads(first_line[len(magic_comment) :])

            indent = conf.pop("indent", "")
            directive_options = conf.pop("jupyter_execute_options", [])

            # don't output the magic comment nor the empty line after it
            if code_cell_lines[1:] and code_cell_lines[1] == "":
                code_cell_lines = code_cell_lines[2:]
            else:
                code_cell_lines = code_cell_lines[1:]

            if len(conf):
                raise ExtensionError(
                    f"Unexpected key(s) in the rst-json-conf: `{conf}` while "
                    f"processing the cell:\n\n{cell_source_code}",
                    modname=__name__,
                )

    return indent, directive_options + [""] + code_cell_lines


# pylint: disable=unused-argument
def make_jupyter_sphinx_block(cell_source_code: str, rst_indent: str = "    ") -> str:
    """
    Converts a code cell into rst code under a :code:`jupyter-execute` directive.

    Indentation is applied according to the magic comment.

    .. note::

        The contents of the :code:`jupyter-execute` block require an indentation as
        well. This one can be set in the :code:`conf.py`.

        E.g., :code:`notebook_to_jupyter_sphinx_rst_indent = "    "`.

    Parameters
    ----------
    cell_source_code
        String containing the code of the cell.
    rst_indent
        Indentation used to indent the code inside the :code:`.. jupyter-execute ::`
        block.
    """
    indent, lines = get_code_indent_and_processed_lines(cell_source_code)
    out = ""
    header = f"\n\n\n{indent}.. jupyter-execute::\n"
    indent = f"{indent}{rst_indent}"
    for line in lines:
        out += f"{indent}{line}\n" if line != "" else "\n"

    return header + out if out.strip() != "" else ""


# pylint: disable=unused-argument
def make_rst_block(cell_source: str, prefix="\n\n\n") -> str:
    """
    Prefixes the raw rst with the :code:`prefix`.

    Parameters
    ----------
    cell_course
        String containing the contents of the raw cell.
    prefix
        Prefix to add to :code:`cell_source`.
    """
    return prefix + cell_source


def cell_to_rst_str(
    cell: dict, is_first_cell: bool = False, rst_indent: str = "    "
) -> str:
    """
    Converts a notebook cell dict according to its type (raw or code).

    Parameters
    ----------
    cell
        Cell dict object from the notebook file.
    is_first_cell
        Indicates if it is the first cell in the notebook file.
        Used to avoid inserting undesired blank lines.
    rst_indent
        See :func:`~.make_jupyter_sphinx_block`.
    """
    cell_type = cell["cell_type"]
    cell_source = cell["source"]

    if cell_type == "code":
        rst = make_jupyter_sphinx_block(cell_source, rst_indent)
    elif cell_type == "raw":
        rst = make_rst_block(cell_source, prefix="" if is_first_cell else "\n\n\n")
    else:
        logger.debug(
            f"Cell of type {cell_type} are ignored. "
            "Only code and raw cells will be processed.",
        )
        rst = ""

    return rst


def notebook_to_rst(notebook: dict, rst_indent: str = "    ") -> str:
    """
    Converts the notebook to an rst string.

    Parameters
    ----------
    notebook
        Dict(-like) object of the notebook file.
    rst_indent
        See :func:`~.make_jupyter_sphinx_block`.
    """
    rst_str = (
        ".. DO NOT EDIT, CHANGES WILL BE LOST!\n"
        ".. Automatically generated by the notebook_to_jupyter_sphinx sphinx extension.\n\n"
    )
    for i, cell in enumerate(notebook["cells"]):
        logger.debug(f"Processing cell #{i}.")
        rst_str += cell_to_rst_str(cell, not i, rst_indent)

    if rst_str[-1] != "\n":
        rst_str += "\n"

    return rst_str


# pylint: disable=unused-argument
def notebooks_to_rst(app, config) -> None:
    """
    Searches for all :code:`*.rst.py` files and converts them to :code:`*.rst` files.

    The output file will placed in the same directory as the original file.

    Parameters
    ----------
    app
        The sphinx app provided by sphinx when calling this function.
    config
        The sphinx config provided by sphinx when calling this function.
    """
    encoding = "utf-8"

    def _write_required(filepath: Path, text: str) -> bool:
        """
        Writes the contents to the file only if these contents will results in a
        different file.
        """
        write = True
        filepath = Path(filepath)
        if filepath.is_file():
            old_contents = filepath.read_text(encoding=encoding)
            if old_contents == text:
                # Avoid making sphinx consider that the file has changed
                write = False

        return write

    srcdir = Path(app.srcdir)
    for file in srcdir.rglob("*.rst.py"):
        logger.debug("Converting file...", location=file)
        try:
            notebook = jupytext.read(file, fmt="py:percent")
            rst_filepath = file.parent / f"{Path(file.stem).stem}.rst"
            rst_indent = config["notebook_to_jupyter_sphinx_rst_indent"]
            always_rebuild = config["notebook_to_jupyter_sphinx_always_rebuild"]
            rst_str = notebook_to_rst(notebook, rst_indent)
            if always_rebuild or _write_required(rst_filepath, rst_str):
                Path(rst_filepath).write_text(rst_str, encoding=encoding)
        except Exception as e:
            raise ExtensionError(  # pylint: disable=raise-missing-from
                f"Unexpected error occurred while converting \n{file}.\n\n", orig_exc=e
            )


def setup(app):
    """
    Setup the sphinx extension by connecting the converter to one of the events that
    sphinx emits in the beginning of the docs build execution.
    """
    app.connect("config-inited", notebooks_to_rst)
    # Register the extension configuration parameters
    app.add_config_value(
        name="notebook_to_jupyter_sphinx_rst_indent",
        default="    ",
        rebuild="html",
        types=[str],
    )
    # Will force the extensions to always write the rst output even if the file would be
    # the same, this should trigger sphinx in processing the file again, which can be
    # useful if the source code used in the notebook has changed
    app.add_config_value(
        name="notebook_to_jupyter_sphinx_always_rebuild",
        default=False,
        rebuild="html",
        types=[bool],
    )
