import os
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version as pkg_version


# -- Path setup --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# -- Project information -----------------------------------------------------
project = "microlux"
author = "microlux contributors"

try:
    release = pkg_version("microlux")
except PackageNotFoundError:  # fallback if not installed
    release = "0.0.0"
# Also provide short X.Y version for Sphinx metadata
version = release


# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Parse Markdown files with MyST
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Autodoc / Autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
}
autodoc_mock_imports = [
    "jax",
    "jax.numpy",
    "VBBinaryLensing",
    "MulensModel",
    "numpyro",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True


# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
