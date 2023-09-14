# Contributing

Contributions are welcome and greatly appreciated! Every little bit helps, and credit will always be given.

In order to contribute to the documentation and/or code please follow the {ref}`Setting up for local development` instructions.

You can join our public [Slack workspace through this link](https://join.slack.com/t/quantify-hq/shared_invite/zt-1nd78r4e9-rbWdna53cW4DO_YbtMhVuA)!
Here you'll find the [software-for-users](https://quantify-hq.slack.com/archives/C01ETDK6P97) and the [software-for-developers](https://quantify-hq.slack.com/archives/C02DE4ZENNQ) channels.

You are also welcome to join our public weekly developer's meeting for which the meeting link is posted in the [software-for-developers](https://quantify-hq.slack.com/archives/C02DE4ZENNQ) channel.

## GitLab issues and merge requests

- [GitLab issues](https://gitlab.com/quantify-os/quantify-core/-/issues)
- [GitLab merge requests](https://gitlab.com/quantify-os/quantify-core/-/merge_requests)
- [Gitlab issue board](https://gitlab.com/groups/quantify-os/-/boards)

## Types of contributions

You can contribute in many ways:

### Reporting of bugs and defects

A defect is any variance between actual and expected result, this can include bugs in the code or defects in the documentation or visualization.

Please report defects to [the GitLab Tracker](https://gitlab.com/quantify-os/quantify-core/-/issues)
using the **Defect** description template.

[Merge Request Guidelines](#merge-request-guidelines) for details on best developmental practices.

### Features

If you wish to propose a feature, please file an issue on [the GitLab Tracker](https://gitlab.com/quantify-os/quantify-core/-/issues) using the **Feature** description template. Community members will help refine and design your idea until it is ready for implementation.
Via these early reviews, we hope to steer contributors away from producing work outside of the project boundaries.

Please see the [Merge Request Guidelines](#merge-request-guidelines) for details on best developmental practices.

### Documentation

Quantify could always use more documentation, whether as part of the official Quantify
docs, in docstrings, tutorials and even on the web in blog posts, articles and such.

`quantify-core` documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/),
and written in MyST Markdown flavour.
MyST is a superset of Markdown that can be mapped directly to reStructuredText.
It provides the same features as reStructuredText, but its Markdown-like syntax
is usually considered more convenient.
Refer to [`myst-parser` documentation](https://myst-parser.readthedocs.io/) for
the details of MyST syntax and user guide.

Docstrings have to be still written in reStructuredText, because parsing them from MyST
is not yet supported in `myst-parser`.
Use the [numpy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
for them.
The only exception is that parameter's type(s) should not be specified in the docstrings
but instead by using [type hints](https://docs.python.org/3/library/typing.html).

## Code style

We are using several tools that help us ensure consistent code style and code quality.

### [Ruff](https://beta.ruff.rs/docs/)

`ruff` is our current primary linter.
It is configured using `pyproject.toml`, refer to it for the exact enabled rule set.

If `ruff` rule needs to be locally suppressed, you can do it either globally or per file using `pyproject.toml`,
or locally using `# noqa: RULE_ID` comments.

### [Black](https://black.readthedocs.io/en/stable/index.html)

We use `black` as our code formatter.
Before commiting, we suggest calling `black .` in the root of the repository to ensure that the code is correctly formatted.

If a part of the code is not formatted nicely with black (for example, if it contains a matrix with manually aligned columns),
`black` [can be locally suppressed](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#code-style)
by wrapping the desired section with `# fmt: off` and `# fmt: on`.

### [Pyright](https://github.com/microsoft/pyright)

We use `pyright` to ensure typing consistency along the code.
It is generally suggested to write strictly typed code and use `isinstance()` checks if needed
to utilize [`pyright`'s type narrowing](https://microsoft.github.io/pyright/#/type-concepts-advanced),
but if you really need to suppress type checks locally, use `# pyright: ignore` magic comments.

### [Pylint](https://pylint.org/) and [Codacy](https://www.codacy.com/)

We are still using Pylint and Codacy, but aiming to fully replace them with `ruff`.
If the file is passing `ruff`, you may fully disable `pylint` on this file by adding the file to ignore list in `.pylintrc`.

### [pre-commit](https://pre-commit.com/)

`pre-commit` helps to always run necessary tools (in our case `ruff` and `black`) before commiting.
We recommend setting it up by calling `pre-commit install` once in the root of the repository.
To execute manually, call `pre-commit run --all-files`.

A known pitfall when using `pre-commit` is that you must always activate a correct virtual environment
in your shell or IDE before calling `git commit` to make `black` and `ruff` executables available.
If you need to disable the pre-commit hooks, simply call `pre-commit uninstall`.

### Suggested IDE configuration

We recommend to configure your IDE to assist you with writing code that respects our code style.
For example, if you are using Visual Studio Code, we recommend to set `python.analysis.typeCheckingMode` to `basic`,
`python.formatting.provider` to `black`, `editor.formatOnSave` to `true` and to enable
[official Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).

## Working on issues

After an issue is created, the progress of the issues is tracked on the [GitLab issue board](https://gitlab.com/groups/quantify-os/-/boards).
The maintainers will update the state using [labels](https://gitlab.com/groups/quantify-os/-/labels) .
Once an issue is ready for review a Merge Request can be opened.

### Issues workflow

The workflow of the issues is managed using the `State | <state>` labels that specifies the current state of the issue. These are intended to be used for issues only, and in *exceptional* cases for Merge Requests that have no associated issue.

An issue does not need to go through all the steps. But we differentiate this many to accommodate for the more nuanced cases.

*Progress captain* denotes the person who is ultimately responsible if any progress is to be made on a specific issue or Merge Request. Note that it can be as simple as asking someone for help/review/merge, or doing these requests again (we are all bandwidth-limited).

```{tip}
Hovering the mouse over each label in GitLab will show these descriptions.
```

- `State | 1. Needs refinement`
  : - *Progress captain*: creator of the issue.
    - The problem in the new issue is not reproducible and/or not clear enough to allow for the design of a solution.
- `State | 2. Workable (design)`
  : - *Progress captain*: None.
    - The problem is recognized and understood. Anyone can pick up the issue. Issues with critical priority will be prioritized by maintainers.
- `State | 3. In design...`
  : - *Progress captain*: assignee & creator of the issue/project owners/maintainers/developers/community.
    - The relevant parties are working to propose and converge on a particular solution/implementation. Issue must have at least one assignee.
- `State | 4. Workable (implementation)`
  : - *Progress captain*: None.
    - The solution is clear. Anyone can pick up the issue implementation.
- `State | 5. In progress...`
  : - *Progress captain*: assignee.
    - The current assignee is actively working on fixing the issue according to the agreed-upon solution and creating a Merge Request.

## Merge request guidelines

Please make merge requests into the *main* branch. Each request should be self-contained and address a single issue on the tracker.

Before you submit a merge request, check that it meets these guidelines:

1. New code should be fully tested; running pytest in coverage mode can help identify gaps.

2. Documentation is updated, this includes docstrings and any necessary changes to existing tutorials, user documentation and so forth. See [Documentation] for docstrings format.

3. The CI pipelines should pass for all merge requests.

```{note} Check the status of the CI pipelines, the status is also reported in the merge request:
- [black](https://github.com/psf/black) formatter should pass (we use default settings).
- The test suite passes.
- Any reasonable code-quality issues raised by [pylint](https://pylint.readthedocs.io/en/latest/index.html) should be addressed.
- No degradation in code coverage.
- Documentation should build.
```

4. Ensure your merge request contains a clear description of the changes made and how it addresses the issue. If useful, add a screenshot to showcase your work to facilitate an easier review. There is a template that you can use when creating a new merge request that you can select in the GitLab interface.

5. Make sure to keep selected the checkbox `Allow commits from members who can merge to the target branch`. This allows maintainers to [collaborate across forks](https://docs.gitlab.com/ee/user/project/merge_requests/allow_collaboration.html) for fine tunning and small fixes before the merge request is accepted.

Congratulations! The maintainers will now review your work and suggest any necessary changes.
If no changes are required, a maintainer will "approve" the merge request.
When your merge request is approved, feel free to add yourself to the list of contributors.
Thank you very much for your hard work in improving quantify!

``````{tip} [Maintainers and developers] In order to commit and push to the original branch of the merge request, you will need:
```
$ # 1. Create and checkout a local branch with the changes of the merge request
$ git fetch git@gitlab.com:thedude/awesome-project.git update-docs
$ git checkout -b thedude-awesome-project-update-docs FETCH_HEAD

$ # 2. Make changes and commit them

$ # 3. Push to the forked project
$ git push git@gitlab.com:thedude/awesome-project.git thedude-awesome-project-update-docs:update-docs
```

N.B. You might need to adapt the `fetch` and `push` commands if you are using `https` instead of `ssh`.
``````

### Merge requests workflow

The workflow of the Merge Requests (MRs) is managed using the `MR State | <state>` labels that specifies the current state of the MR as described below, and the *progress captain* denotes the same as in the [Issues workflow].

```{tip}
Hovering the mouse over each label in GitLab will show these descriptions.
```

- `MR State | 1. In progress...` *Progress captain*: assignee.
    - MR not ready for complete review. Equivalent to Draft/WIP. The assignee is responsible for asking help/advice by tagging relevant people.
    - Next state: `2. Review me!`.
- `MR State | 2. Review me!` *Progress captain*: assignee.
    - MR was submitted and is ready for review. Assignee may tag potential reviewers in the comments. Next state: "3. In review...".
    - Next state: `3. In review...`.
- `MR State | 3. In review...` *Progress captain*: reviewer.
    - A reviewer with enough expertise is reviewing the MR (the reviewer should self-assign as such). If there are no concerns so far and the reviewer does not have enough expertise, the `2. Review me!` label should be activated again.
    - Next state: `4. Change requested` or `5. Merge me!`.
- `MR State | 4. Change requested` *Progress captain*: assignee.
    - Reviewer's comments need to be addressed (comments/code/test/docs/etc.). Conflict with target branch should be addressed carefully.
    - Next state: `2. Review me!`.
- `MR State | 5. Merge me!` *Progress captain*: assignee & maintainer.
    - MR ready to be merged. Assignee should tag maintainers.
    - Next state: Merged or `4. Change requested`.

When moving the MRs between states, the next *progress captain* should be tagged in the comments. This is the only reliable way for them to get notified.

### Versioning, backward compatibility and deprecation policy

```{note}
This policy is valid from the `1.0` release of `quantify-core` and any project that adopts it.
```

We adopt a semantic versioning scheme for all subprojects of Quantify.
Version numbers consist of three numbers: a major, minor and patch version consequently.
A major version bump is performed, when the API of a project has changed significantly
and requires a rewrite of significant parts of user code.
A minor version bump is done for the new features, and a patch version bump is for bugfixes in the released version.

We aim to provide backward compatibility for the future releases of Quantify within the same major version.
This means that code that uses `quantify-core-1.1` should be also executable using `quantify-core-1.2`.
It is not realistic to keep this policy for *every* minor breaking change,
for example renaming of a method or a change of its location.
For such minor changes, we may deprecate some parts of API with a warning and remove it after three minor version releases.
Therefore, we define the following policies:

1. If a function is a part of the public API, it should be declared in the ``__all__`` list of a module.
2. We guarantee backwards-compatible behaviour for all public API elements for three minor releases forward,
   unless this element is deprecated.
3. Each deprecation must issue a ``FutureWarning``, mentioning the release when this part of the API is dropped
   and if possible, a (short) explanation on how to port the code.
4. Violation of backwards compatibility is considered an issue
   and a bugfix release fixing it must be done as soon as possible.
5. If some part of the API is not declared as public, but is important for some use-cases,
   a user can (and is encouraged) to request stabilizing it using an issue.
   If this proposal is accepted, this part of the API should be stabilized in the next minor release.

For example, if some function is deprecated in the `quantify-core-1.2` release,
it should be marked for removal in the `quantify-core-1.5` using a ``FutureWarning``,
mentioning the version of removal (in this case `1.5`)
and an instruction for the user on how to port the code to the new versions of Quantify.
The recommended way to do it is through the `quantify_core.utilities.deprecated` decorator.

If there is doubt about whether the API change is considered major (requiring major version bump) or minor,
it must be discussed during a developers meeting.
Changes to this policy should also be discussed and approved during a developers meeting.
