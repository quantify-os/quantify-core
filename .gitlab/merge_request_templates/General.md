### Explanation of changes

Describe the contents of this merge request and the issues addressed.
Add screenshots if this helps your explanation.

### Motivation of changes

Motivate why the particular solution was chosen.

--------------------

## Merge checklist
See also [merge request guidelines](https://quantify-quantify-core.readthedocs-hosted.com/en/latest/contributing.html#merge-request-guidelines)

- [ ] Merge request has been reviewed and approved by a project maintainer.
- [ ] Merge request contains a clear description of the proposed changes and the issue it addresses.
- [ ] Merge request made onto appropriate branch (develop for most MRs).
- [ ] New code is fully tested.
- [ ] New code is documented and docstrings use [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html).
- [ ] CI pipelines pass
    - [ ] black code-formatting passes (gitlab-ci),
    - [ ] test suite passes (gitlab-ci),
    - [ ] no degradation in code-coverage (codacy),
    - [ ] no (serious) new pylint code quality issues introduced (codacy),
    - [ ] documentation builds successfully (CI and readthedocs).
