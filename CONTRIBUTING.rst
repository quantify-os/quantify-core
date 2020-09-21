.. highlight:: shell

============
Contributing
============

Contributions are welcome and greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Bugs
~~~~

Please report bugs to [the GitLab Tracker](https://gitlab.com/qblox/packages/software/quantify/issues)
using the **Bug** description template.

Issues on the Tracker marked with the *Bug* or *Help Wanted* labels are open to contributions. Please see the
[Merge Request Guidelines](#merge-request-guidelines) for details on best developmental practices.

Features
~~~~~~~~

If you wish to propose a feature, please file an issue on [the GitLab Tracker](https://gitlab.com/qblox/packages/software/quantify/issues)
using the **Enhancement** description template. Community members will help refine and design your idea until it is
ready for implementation. Via these early reviews, we hope to steer contributors away from producing work outside of
the project boundaries.

Issues on the Tracker marked with the *Enhancement* or *Help Wanted* labels are ready and open to contributions.
Please see the [Merge Request Guidelines](#merge-request-guidelines) for details on best developmental practices.

Documentation
~~~~~~~~~~~~~

quantify could always use more documentation, whether as part of the official quantify docs, in docstrings, tutorials
and even on the web in blog posts, articles and such. Please follow the [bugs](#bugs) workflow when contributing
documentation changes.

For docstrings, please use the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Setting up quantify for local development
------------------------------------------------

Ready to contribute? Here's how to set up `quantify` for local development.

1. Fork the `quantify` repo on GitLab.
2. Clone your fork locally::

    $ git clone git@gitlab.com:your_name_here/quantify.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv quantify
    $ cd quantify/
    $ pip install -e .
    $ pip install -r requirements_dev.txt

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 the tests and have test coverage::

    $ flake8 quantify tests
    $ pytest --cov

  If you have worked on documentation instead of code you may want to preview how your docs look locally.
  You can build the docs locally using:

  .. code-block:: shell

      $ cd docs
      $ make html

  The docs will be located in `quantify/docs/_build`.

6. Commit your changes and push your branch to GitLab::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a merge request through the GitLab website.

Merge Request Guidelines
--------------------------

Please make merge requests into the *develop* branch (not the *master* branch). Each request should be self-contained and address a single issue on the tracker.

Before you submit a merge request, check that it meets these guidelines:

1. New code should be fully tested; running pytest in coverage mode can help identify gaps.
2. Documentation is updated, this includes docstrings and any necessary changes to existing tutorials, user documentation and so forth. We use the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
3. The CI pipelines should pass for all merge requests.

   - Check the status of the pipelines https://gitlab.com/qblox/packages/software/quantify/pipelines, the status is also reported in the merge request.
   - flake8 linter should pass.
   - No degradation in code coverage.
   - Documentation should build.

Congratulations! Community members will now review your work and suggest any necessary changes. Thank you very much
for your hard work in improving quantify.
