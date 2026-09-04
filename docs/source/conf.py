# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import version as pkg_version

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "pylinkage"
copyright = "2021, Hugo Farajallah"
author = "Hugo Farajallah"

# The full version, including alpha/beta/rc tags
release = pkg_version("pylinkage")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Use docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # Parse Google-style docstrings ("Args:", "Returns:", "Raises:").
    # Without this they are read as definition lists, and "*args"/"**kwargs"
    # are read as emphasis markup.
    "sphinx.ext.napoleon",
    # Useful for markdown integration
    "myst_parser",
    "sphinx.ext.githubpages",
]

# Render "Attributes:" as :ivar: fields rather than standalone .. attribute::
# directives. Without this, a documented attribute collides with the entry
# autodoc already generates for the same dataclass field.
# Generate anchors for h1-h3 headings, so in-page README links such as
# "[tutorials](#tutorials)" resolve here as they do on GitHub.
myst_heading_anchors = 3

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Files to be used as source
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["tests"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
