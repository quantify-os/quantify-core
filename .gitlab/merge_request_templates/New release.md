## Checklist for a new release

1. [ ] Review `CHANGELOG.md` and `AUTHORS.md` have been updated.
1. [ ] Update `Unreleased` chapter title in `CHANGELOG.md` to `X.Y.Z (YYYY-MM-DD)`. Commit it.
1. [ ] Review `@deprecated` and `FutureWarnings` that can be cleaned up now.

1. CI pipeline:
    - [ ] Automated pipeline passes.
    - [ ] All `Test (py3.x, Windows, manual)` pass (trigger manually!).

1. [ ] Commit pip frozen requirements for future reference:
    - Go to the `Test (py3.9, Linux)` pipeline job and download the `artifacts` (right side "Job artifacts" `-->` "Download").
    - Unzip, get the `frozen-requirements.txt`.
    - Paste it in `frozen-requirements` directory.
    - Rename it, commit & push:

      ```bash
      NEW_VERSION=X.Y.Z  # Can also be X.Y.Z.rcT for release candidate
      echo $NEW_VERSION

      mv frozen-requirements.txt frozen-requirements-$NEW_VERSION.txt

      git add ./frozen_requirements/frozen-requirements-$NEW_VERSION.txt
      git commit -m "Add pip frozen requirements for $NEW_VERSION"
      git push
      ```

1. [ ] Create tag for bumped version:
    - Merge this MR into `main`.
    - Create and push an **annotated** tag `vX.Y.Z` pointing to the merge commit:

      ```bash
      echo $NEW_VERSION

      git tag -a "v${NEW_VERSION}"  # Note: should be vX.Y.Z, not X.Y.Z
      # You will be prompted for a tag description here. Provide a list of highlights.
      git push origin "v${NEW_VERSION}"
      ```
    <!-- - Future TODO: finish automation of this step in `.gitlab-ci.yml`. -->
    <!-- 1. [ ] Run **one** of the major/minor/patch version bump (manual) jobs in the CI pipeline of the MR. -->
    <!--     - NB this can only be done after unix and windows test & docs jobs pass. -->

1. [ ] Add `Unreleased` chapter back to `CHANGELOG.md`. Commit and push it to `main` directly (no need to review it). Commit message could be, for example, `start development of vX.Y.(Z+1)`.

1. [ ] Create and push an **annotated** tag `vX.Y.(Z+1).dev` pointing to the commit above.  Commit annotation could be, for example, `Start development of vX.Y.Z+1`.
    <!-- Note: if we are following semver, this should be rather vX.(Y+1).0.dev, and bugfixes need to go into a separate bugfix branch for each minor release a-la `stable/vX.Y`.
    Since we are not so strict with that and releasing minor and bugfix from the same branch, to avoid situation of having previous commit having version v0.7.0.dev19+abcdef and
    next commit version v0.6.5 (which is less than v0.7.0.devN, which should not be the case) we must bump a bugfix version (the most minor version we bump in main)
    and later we may decide that we are releasing a minor instad of a bugfix.-->

1. Read-the-Docs setup:
   - [ ] Enable docs build for the new tag [over here](https://readthedocs.com/projects/quantify-quantify-core/versions/).
      - Configuration:
        - `Active`=True
        - `Hidden`=False
        - `Privacy Level`=Public
   - [ ] Change both the `Default version` and `Default branch` of the docs to the tag that was released [over here](https://readthedocs.com/dashboard/quantify-quantify-core/advanced/). Hit Save!
   - [ ] Make sure the docs build and check on RTD.
      - Manually rebuild `latest` by hitting `Build version:` [over here](https://readthedocs.com/projects/quantify-quantify-core/builds/).
      - Check both the `latest` and the new version links on RTD work by clicking through to Changelog (hit Ctrl+F5).

1. [ ] Create [new release on GitLab](https://gitlab.com/quantify-os/quantify-core/-/releases).
    - Meaningful title
    - List of highlights followed by changelog.
    - Add a few images or animated GIFs showcasing the new exciting features.

1. When `Release to test.pypi.org` job of the tag pipeline succeeds:
    - [ ] Install package in (test) env and validate (e.g., run a quick notebook).
       ```bash
       pip install quantify-core==x.x.x --extra-index-url=https://test.pypi.org/simple/
       ```
       - _(For creating test env)_
         ```bash
         ENV_NAME=qtest # Adjust
         PY_VER=3.8
         DISPLAY_NAME="Python $PY_VER Quantify Test Env" && echo $DISPLAY_NAME # Adjust

         conda create --name $ENV_NAME python=$PY_VER
         conda activate $ENV_NAME
         conda install -c conda-forge jupyterlab
         python -m ipykernel install --user --name=$ENV_NAME --display-name="$DISPLAY_NAME"
         ```

1. [ ] Release on PyPi by triggering manual `Release to pypi.org` job and wait till it succeeds.
1. [ ] Post the new release in Slack (`#software-for-users` and `#software-for-developers`).
    - PS Rockets are a must! 🚀🚀🚀
1. [ ] Inform the Quantify Marketing Team.
