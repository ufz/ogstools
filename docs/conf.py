# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime

import ogstools

project = "ogstools"
author = ogstools.__authors__
copyright = f"{datetime.now().year}, {author}"
version = release = ogstools.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinxarg.ext",
    "sphinxcontrib.programoutput",
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "reference/modules.rst",
    "reference/ogstools.rst",
]

myst_enable_extensions = ["dollarmath", "colon_fence", "amsmath"]
myst_heading_anchors = 3
# myst_title_to_header = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["ogstools.css"]
html_context = {"default_mode": "light"}

html_theme_options = {
    "logo": {
        "image_light": "logo.png",
        "image_dark": "logo.png",
    },
    # "announcement": "<p style='font-weight: 400'>Some announcement</p>",
    "icon_links": [
        {
            "name": "GitLab",
            "url": "https://gitlab.opengeosys.org/ogs/tools/ogstools",
            "icon": "fa-brands fa-square-gitlab",
            "type": "fontawesome",
        }
    ],
}

show_authors = True

# sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False
