 ## Checklist for a new release

- [ ] Review `CHANGELOG.rst` and `AUTHORS.rst` have been updated.
- [ ] Review deprecation warnings that can be cleaned up now.
- [ ] CI pipelines pass
    - [ ] windows tests pass (trigger them manually!).
- [ ] Save the pip dependencies for future reference:
    - [ ] Go to the `python3.8` tests pipeline job and download the `artifacts` (right side "Job artifacts" `-->` "Download").
    - [ ] Unzip, get the `frozen-requirements.txt`.
    - [ ] Move it to `frozen-requirements` directory.
    - [ ] Rename it, commit & push:

```bash
NEW_TAG=$(git describe --tags --abbrev=0)
echo $NEW_TAG
mv frozen-requirements.txt frozen-requirements-$NEW_TAG.txt
git add ./frozen_requirements/frozen-requirements-$NEW_TAG.txt
git commit -m "Add pip frozen requirements for $NEW_TAG"
```
<!-- - [ ] Run **one** of the major/minor/patch version bump (manual) jobs in the CI pipeline of the MR. -->
<!--     - NB this can only be done after unix and windows test & docs jobs pass. -->

- [ ] Bump version and push new tag:
    - Future TODO: finish automation of this step in `.gitlab-ci.yml`.

```bash
# Version bump
VERSION_PART=patch # or minor, or major
bump2version $VERSION_PART --config-file setup.cfg
git push --follow-tags
```

- [ ] "Activate" the RTD docs build for the new tag [over here](https://readthedocs.com/projects/quantify-quantify-core/versions/).
    - Configuration:
        - `Active`=True
        - `Hidden`=False
        - `Privacy Level`=Public
- [ ] Change the "Default version" of the docs to the tag that was released [over here](https://readthedocs.com/dashboard/quantify-quantify-core/advanced/).
- [ ] Make sure the docs build.
- [ ] Create [new release on GitLab](https://gitlab.com/quantify-os/quantify-core/-/releases).
    - [ ] Meaningful title
    - [ ] List of highlights followed by changelog.
    - [ ] Add a few images or animated GIFs showcasing the new exciting features.

- [ ] Test PyPi release (see also https://adriaanrol.com/posts/pypi/).
- [ ] Release on PyPi and wait for it to become available (see also https://adriaanrol.com/posts/pypi/).
- [ ] Post the new release in Slack (`#software-for-users` and `#software-for-developers`).
    - PS Rockets are a must! 🚀🚀🚀
- [ ] Inform the Quantify Marketing Team.
