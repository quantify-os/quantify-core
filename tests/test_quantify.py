import pytest
import os
import re
import datetime
import pprint
from pathlib import Path


def test_copyrights():
    skip_files = {'__init__.py'}
    failures = {}
    quantify_path = Path(__file__).resolve().parent / os.path.join("..", "quantify")
    current_year = str(datetime.datetime.now().year)
    for root, dirs, files in os.walk(quantify_path):
        path = os.path.relpath(root, quantify_path).split(os.sep)
        for file in files:
            if file[-3:] == '.py' and file not in skip_files:
                filepath = os.path.join(*path, file)
                with open(os.path.join(quantify_path, filepath), 'r') as f:
                    copyright_found = False
                    for line in f.readlines():
                        if re.match(".*Copyright.*Qblox BV", line):
                            copyright_found = True
                            if current_year not in line:
                                failures[filepath] = "No copyright claim for {}.".format(current_year)
                            break
                    if not copyright_found:
                        failures[filepath] = "No copyright found."
    if failures:
        pytest.fail("Bad copyrights:\n{}".format(pprint.pformat(failures)))
