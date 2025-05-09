# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import warnings
from datetime import datetime

import pyvista
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper
from sphinx.deprecation import RemovedInSphinx90Warning
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

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
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
    "sphinxcontrib.apidoc",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.programoutput",
    "myst_nb",
    "pyvista.ext.viewer_directive",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]


templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "reference/modules.rst",
    "reference/ogstools.rst",
    "examples/**/README.rst",
    "examples/README.rst",
    "user-guide/README.rst",
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

# Set up the version switcher.  The versions.json is stored in the doc repo.
if os.environ.get("CI_MERGE_REQUEST_IID", False):
    switcher_version = f"!{os.environ['CI_MERGE_REQUEST_IID']}"
elif ".dev" in version:
    switcher_version = "dev"
else:
    switcher_version = f"{version}"

html_theme_options = {
    "logo": {
        # "text": "OGSTools",
        "image_light": "ogstools.png",
        "image_dark": "ogstools-dark.png",
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
    "navigation_with_keys": True,
    "switcher": {
        "version_match": switcher_version,
        "json_url": "https://ogstools.opengeosys.org/_static/versions.json",
    },
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "show_version_warning_banner": True,
}

nitpick_ignore_regex = [("py:class", r".*")]

show_authors = True

# sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

### apidoc / autodoc settings ###
apidoc_module_dir = "../ogstools"
apidoc_output_dir = "reference"
apidoc_excluded_paths = [
    "../**/examples/**",
    "../**/user-guide/**",
    "./**/tests/**",
    "../**/**/templates/**",
]
apidoc_separate_modules = True
apidoc_module_first = True
apidoc_extra_args = ["--force", "--implicit-namespaces"]

autoclass_content = "both"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"

autodoc_default_options = {"special-members": "__call__, __getitem__"}

### sphinx-gallery setup ###
# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True

# Disable progress bars in sphinx
os.environ["TQDM_DISABLE"] = "1"


def reset_plot_setup(*_):
    "Reset the ogstools plot setup to its default values"

    ogstools.plot.setup.reset()


sphinx_gallery_conf = {
    "examples_dirs": ["examples", "user-guide"],
    "gallery_dirs": ["auto_examples", "auto_user-guide"],
    "show_signature": False,
    "download_all_examples": False,
    "image_scrapers": ("matplotlib", DynamicScraper()),
    "matplotlib_animations": True,
    "reset_modules": ("matplotlib", reset_plot_setup),
    "subsection_order": ExplicitOrder(
        [
            "examples/howto_quickstart",
            "examples/howto_preprocessing",
            "examples/howto_prjfile",
            "examples/howto_logparser",
            "examples/howto_postprocessing",
            "examples/howto_plot",
            "examples/howto_conversions",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
}

# feflowlib is optional
try:
    import ifm_contrib as ifm  # noqa: F401
except ImportError:
    print("Skipping FEFLOW-related functionality for the documentation!")
    exclude_patterns.extend(
        ["reference/ogstools.feflowlib.*", "user-guide/feflowlib*"]
    )
    apidoc_excluded_paths.append("../**/feflowlib/**")
    apidoc_excluded_paths.append("../docs/releases/ogstools-*.md")
    sphinx_gallery_conf["ignore_pattern"] = r".*_feflowlib_*"


suppress_warnings = ["config.cache"]

warnings.filterwarnings("ignore", category=RemovedInSphinx90Warning)


# Suppress sphinx warning for multiple documents generated by sphinx-gallery
# May break myst-nb?
def setup(app):
    app.registry.source_suffix.pop(".ipynb", None)
