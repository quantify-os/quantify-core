.. highlight:: shell

============
Contributing
============

Contributions are welcome and greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Reporting of Bugs and Defects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A defect is any variance between actual and expected result, this can include bugs in the code or defects in the documentation or visualization.

Please report defects to the `the GitLab Tracker <https://gitlab.com/quantify-os/quantify-core/-/issues>`_
using the **Defect** description template.

`Merge Request Guidelines`_ for details on best developmental practices.

Features
~~~~~~~~

If you wish to propose a feature, please file an issue on `the GitLab Tracker <https://gitlab.com/quantify-os/quantify-core/-/issues>`_ using the **Feature** description template. Community members will help refine and design your idea until it is ready for implementation.
Via these early reviews, we hope to steer contributors away from producing work outside of the project boundaries.

Please see the `Merge Request Guidelines`_ for details on best developmental practices.

Documentation
~~~~~~~~~~~~~

quantify could always use more documentation, whether as part of the official quantify docs, in docstrings, tutorials and even on the web in blog posts, articles and such.

For docstrings, please use the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Working on issues
------------------

After an issue is created, the progress of the issues is tracked on the `GitLab issue board <https://gitlab.com/quantify-os/quantify-core/-/boards>`_.
The maintainers will update the state using `labels <https://gitlab.com/quantify-os/quantify-core/-/labels>`_ .
Once an issue is ready for review a Merge Request can be opened.



Merge Request Guidelines
--------------------------

Please make merge requests into the *develop* branch (not the *master* branch). Each request should be self-contained and address a single issue on the tracker.

Before you submit a merge request, check that it meets these guidelines:

1. New code should be fully tested; running pytest in coverage mode can help identify gaps.
2. Documentation is updated, this includes docstrings and any necessary changes to existing tutorials, user documentation and so forth. We use the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
3. The CI pipelines should pass for all merge requests.

   - Check the status of the pipelines, the status is also reported in the merge request.
   - flake8 linter should pass.
   - No degradation in code coverage.
   - Documentation should build.
4. Ensure your merge request contains a clear description of the changes made and how it addresses the issue. If useful, add a screenshot to showcase your work to facilitate an easier review.

Congratulations! The maintainers will now review your work and suggest any necessary changes.
If no changes are required, a maintainer will "approve" the review.
If you are
Thank you very much
for your hard work in improving quantify.


Setting up quantify for local development
------------------------------------------------

Ready to contribute? Here's how to set up `quantify` for local development.

1. Fork the `quantify` repo on GitLab.
#. Clone your fork locally::

    $ git clone git@gitlab.com:your_name_here/quantify-core.git

#. Install quantify locally::

    $ cd quantify-core/
    $ pip install -e .
    $ pip install -r requirements_dev.txt

#. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

#. When you're done making changes, check that your changes pass flake8, the tests and have test coverage::

    $ flake8 quantify tests
    $ pytest --cov


  .. tip:: Running parts of the test suite

      To run only parts of the test suite, specify the folder in which to look for
      tests as an argument to pytest. The following example


      .. code-block:: shell

          $ py.test tests/measurement --cov quantify/measurement

      will look for tests located in the tests/measurement directory and report test coverage of the quantify/measurement module.

6. Building the documentation

  If you have worked on documentation instead of code you may want to preview how your docs look locally.
  You can build the docs locally using:

  .. code-block:: shell

      $ cd docs
      $ make html

  The docs will be located in `quantify/docs/_build`.

  .. tip::

      If you are working on documentation it can be useful to automatically rebuild the docs after every change.
      This can be done using the `sphinx-autobuild` package. Through the following command:

      .. code-block:: shell

          $ sphinx-autobuild docs docs/_build/html

      The documentation will then be hosted on `localhost:8000`


7. Commit your changes and push your branch to GitLab::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

#. Submit a merge request through the GitLab website.




