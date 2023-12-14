# Minimal makefile for Sphinx documentation
#

# -vv can be appended below to activate sphinx verbose mode
# For a reference of the different sphinxopts flags,
# see https://www.sphinx-doc.org/en/master/man/sphinx-build.html
SPHINXOPTS    = --keep-going -n -w build_errors.log --color
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = quantify_core
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
