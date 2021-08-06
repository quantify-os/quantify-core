#!/usr/bin/env python3

from pathlib import Path
import click
from quantify_core.utilities._docs_helpers import notebook_to_rst


@click.command()
@click.argument("notebook-filepath", required=True, type=click.Path())
def main(notebook_filepath: Path):
    notebook_filepath = Path(notebook_filepath)
    rst_filepath = notebook_filepath.parent / notebook_filepath.stem
    notebook_to_rst(notebook_filepath, rst_filepath)


if __name__ == "__main__":
    main()
