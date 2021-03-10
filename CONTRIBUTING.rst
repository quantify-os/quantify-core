.. highlight:: console

Contributing
============

Contributions are welcome and greatly appreciated! Every little bit helps, and credit will always be given.

In order to contribute to the documentation and/or code please follow the :ref:`Setting up for local development` instructions.

Types of Contributions
----------------------

You can contribute in many ways:

Reporting of Bugs and Defects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Quantify could always use more documentation, whether as part of the official quantify docs, in docstrings, tutorials and even on the web in blog posts, articles and such.

For docstrings, please use the `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_. The only exception is that parameter's type(s) should not be specified in the docstrings but instead by using `type hints <https://docs.python.org/3/library/typing.html>`_.

Working on issues
------------------

After an issue is created, the progress of the issues is tracked on the `GitLab issue board <https://gitlab.com/quantify-os/quantify-core/-/boards>`_.
The maintainers will update the state using `labels <https://gitlab.com/quantify-os/quantify-core/-/labels>`_ .
Once an issue is ready for review a Merge Request can be opened.

Merge Request Guidelines
------------------------

Please make merge requests into the *develop* branch (not the *master* branch). Each request should be self-contained and address a single issue on the tracker.

Before you submit a merge request, check that it meets these guidelines:

1. New code should be fully tested; running pytest in coverage mode can help identify gaps.
#. Documentation is updated, this includes docstrings and any necessary changes to existing tutorials, user documentation and so forth. See `Documentation`_ for docstrings format.
#. The CI pipelines should pass for all merge requests.

    - Check the status of the CI pipelines, the status is also reported in the merge request.
        - `black <https://github.com/psf/black>`_ linter should pass (we use default settings).
        - The test suite passes.
        - Any reasonable code-quality issues raised by `pylint <https://pylint.readthedocs.io/en/latest/index.html>`_ should be addressed.
        - No degradation in code coverage.
        - Documentation should build.

#. Ensure your merge request contains a clear description of the changes made and how it addresses the issue. If useful, add a screenshot to showcase your work to facilitate an easier review. There is a template that you can use when creating a new merge request that you can select in the gitlab interface.
#. Make sure to keep selected the checkbox `Allow commits from members who can merge to the target branch`. This allows maintainers to `collaborate across forks <https://docs.gitlab.com/ee/user/project/merge_requests/allow_collaboration.html>`_ for fine tunning and small fixes before the merge request is accepted.

Congratulations! The maintainers will now review your work and suggest any necessary changes.
If no changes are required, a maintainer will "approve" the merge request.
When your merge request is approved, feel free to add yourself to the list of contributors.
Thank you very much for your hard work in improving quantify!

.. tip::

    (Maintainers and developers)
    In order to commit and push to the original branch of the merge request, you will need:

    .. code-block::

        # 1. Create and checkout a local branch with the changes of the merge request
        $ git fetch git@gitlab.com:thedude/awesome-project.git update-docs
        $ git checkout -b thedude-awesome-project-update-docs FETCH_HEAD

        # 2. Make changes and commit them

        # 3. Push to the forked project
        $ git push git@gitlab.com:thedude/awesome-project.git thedude-awesome-project-update-docs:update-docs

    N.B. You might need to adapt the `fetch` and `push` commands if you are using `https` instead of `ssh`.

