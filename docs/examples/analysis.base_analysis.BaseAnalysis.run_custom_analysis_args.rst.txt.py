# ---
# jupyter:
#   jupytext:
#     cell_markers: \"\"\"
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
rst_conf = {"jupyter_execute_options": [":hide-code:"]}
# pylint: disable=line-too-long
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement
# pylint: disable=attribute-defined-outside-init
# pylint: disable=duplicate-code


# %% [raw]
"""

.. admonition:: Implementing a custom analysis that requires user input
    :class: dropdown, note

    When implementing your own custom analysis you might need to pass in a few
    configuration arguments. That should be achieved by overriding this
    function as show below.
"""

# %%
rst_conf = {"indent": "    "}

from quantify_core.analysis.base_analysis import BaseAnalysis


# pylint: disable=too-few-public-methods
class MyAnalysis(BaseAnalysis):
    """A docstring for the custom analysis."""

    # pylint: disable=arguments-differ
    def run(self, optional_argument_one: float = 3.5e9):
        """
        A docstring with relevant notes about the analysis execution.

        Parameters
        ----------
        optional_argument_one:
            Explanation of the usage of this parameter
        """
        # Save the value to be used in some step of the analysis
        self.optional_argument_one = optional_argument_one

        # Execute the analysis steps
        self.execute_analysis_steps()
        # Return the analysis object
        return self

    # ... other relevant methods ...
