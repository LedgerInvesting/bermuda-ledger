# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
import bermuda

# -- Project information -----------------------------------------------------

project = "Bermuda"
copyright = f"{date.today().year}, Korra, LLC."
author = "Korra, LLC."
version = bermuda.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_wagtail_theme",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinxext_altair.altairplot",
]

# The suffix(es) of source filenames
source_suffix = [".md", ".rst"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_wagtail_theme"  # "sphinx_rtd_theme"
html_logo = "smbol.png"
html_theme_options = dict(
    project_name="Bermuda",
    logo=html_logo,
    logo_alt="bermuda-logo",
    logo_height=70,
    logo_width=70,
    github_url="https://github.com/LedgerInvesting/bermuda-ledger/blob/main/docs/source/",
    footer_links=[],
)
html_show_sphinx = False
html_sidebars = {
    "**": [
        "searchbox.html",
        "globaltoc.html",
    ]
}
html_favicon = html_logo

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

nbsphinx_execute = "never"
