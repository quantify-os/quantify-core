.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://gitlab.com/qblox/packages/software/quantify/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitLab issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitLab issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

quantify could always use more documentation, whether as part of the
official quantify docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://gitlab.com/qblox/packages/software/quantify/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

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

7. Submit a pull request through the GitLab website.

Merge Request Guidelines
--------------------------

Before you submit a merge request, check that it meets these guidelines:

1. The merge request should include tests.
2. If the merge request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring.
3. The pipelines should pass for all merge requests.

   - Check the status of the pipelines https://gitlab.com/qblox/packages/software/quantify/pipelines, the status is also reported in the merge request.
   - Tests should pass for all versions of python.
   - flake8 linter should pass.
   - pylint should not flag serious issues.
   - Documentation should build.
   - Pipeline should work for PyPy (TODO)

Tips
----

- Ensure you have installed the `requirements_dev.txt`

- To run a subset of tests::

  $ pytest tests.test_quantify

- To auto rebuild docs when editing::

  $ pip install sphinx-autobuild
  $ sphinx-autobuild docs docs/_build/html

  The docs will now be served on http://localhost:8000/



Deploying
---------

.. note::

  TODO Work out how to tag versions using gitlab and deploy to PyPy

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

  $ bump2version patch # possible: major / minor / patch
  - $ git push
  $ git push --tags

Travis will then deploy to PyPI if tests pass.


