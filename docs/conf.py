# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import re
import warnings
from datetime import datetime

# myst-nb and sphinx-autodoc-typehints use deprecated Sphinx API that will be removed
# in Sphinx 10. Suppress until those packages release fixes.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="myst_nb")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="sphinx_autodoc_typehints"
)

import pyvista
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper
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
<<<<<<< HEAD
    "sphinx.ext.autodoc.typehints",
=======
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
>>>>>>> 01f15eed (Fix sphinx warnings)
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
if os.environ.get("CI_MERGE_REQUEST_IID"):
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
    "secondary_sidebar_items": [
        "page-toc",
        "sg_download_links",
        "sg_launcher_links",
    ],
}

# Resolve ambiguous cross-references by preferring top-level API imports
# When Model and Simulation are referenced, prefer ogstools.Model over ogstools.core.model.Model
autodoc_type_aliases = {
    "Model": "ogstools.Model",
    "Simulation": "ogstools.Simulation",
}

# Configure autosummary to prefer the top-level API imports
intersphinx_disabled_reftypes = []

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

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


# Custom function to generate Binder URLs
def custom_gen_binder_url(fpath, binder_conf, _gallery_conf):
    """Generate a Binder URL according to the configuration in conf.py.

    Parameters
    ----------
    fpath: str
        The path to the `.py` file for which a Binder badge will be generated.
    binder_conf: dict or None
        The Binder configuration dictionary. See `gen_binder_rst` for details.

    Returns
    -------
    binder_url : str
        A URL that can be used to direct the user to the live Binder
        environment.
    """
    relpath = re.sub(r".*?examples/", "examples/", str(fpath))
    # Make sure we have the right slashes (in case we're on Windows)
    relpath = relpath.replace(os.sep, "/")

    # Create the URL
    binder_url = "/".join(
        [
            binder_conf["binderhub_url"],
            "v2",
            "gh",
            binder_conf["org"],
            binder_conf["repo"],
            binder_conf["branch"],
        ]
    )
    binder_url += "?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgitlab.opengeosys.org%252Fogs%252Ftools%252Fogstools%26urlpath%3Dlab%252Ftree%252Fogstools/docs/"
    binder_url += relpath

    return binder_url


# Monkey-patching the Sphinx-Gallery binder url
from sphinx_gallery import interactive_example as ie

ie.gen_binder_url = custom_gen_binder_url

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
            "examples/howto_framework",
            "examples/howto_preprocessing",
            "examples/howto_prjfile",
            "examples/howto_simulation",
            "examples/howto_logparser",
            "examples/howto_postprocessing",
            "examples/howto_plot",
            "examples/howto_conversions",
            "*",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
    "binder": {
        "org": "bilke",
        "repo": "binder-ogs-requirements",
        "branch": "6.5.6-0.7.1",  # Always update before release!
        "binderhub_url": "https://binder.opengeosys.org",
        # The following are not used because of monkey patching
        "dependencies": ["./requirements.txt"],
        "notebooks_dir": "notebooks",
        "use_jupyter_lab": True,
    },
}

suppress_warnings = [
    "config.cache",
    # Third-party packages (numpy, PIL) expose internal types that sphinx_autodoc_typehints
    # cannot import; suppress until upstream fixes are available.
    "sphinx_autodoc_typehints.guarded_import",
    # pint's PlainUnit has forward references to 'Unit' which isn't resolvable here.
    "sphinx_autodoc_typehints.forward_reference",
]

nitpick_ignore_regex = [
    # sphinx_autodoc_typehints emits cross-references for typing builtins that
    # Sphinx cannot resolve as inventory entries.
    ("py:data", r"typing\..*"),
    ("py:data", "Ellipsis"),
    # watchdog is a third-party package without intersphinx inventory.
    # sphinx_autodoc_typehints emits both fully-qualified and short names.
    ("py:class", r"watchdog\..*"),
    ("py:class", r"(Dir|File)(Created|Modified)Event"),
]

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
    suppress_warnings.extend(["toc.excluded", "myst.xref_missing"])

# warnings.filterwarnings("ignore", category=RemovedInSphinx90Warning)


def hide_sg_links(app, pagename, _templatename, _context, _doctree):
    if pagename.startswith("auto_examples/"):
        app.add_css_file("hide_links.css")


# Suppress sphinx warning for multiple documents generated by sphinx-gallery
# May break myst-nb?
def setup(app):
    app.registry.source_suffix.pop(".ipynb", None)
    app.connect("html-page-context", hide_sg_links)
