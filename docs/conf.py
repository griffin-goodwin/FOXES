"""Sphinx configuration for the public helio_foxes API and usage guides."""

import inspect
import os
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with (ROOT / "pyproject.toml").open("rb") as project_file:
    metadata = tomllib.load(project_file)["project"]

project = "FOXES"
author = "Griffin Goodwin"
copyright = "2026"
release = metadata["version"]
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.githubpages",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
# Local reprocessing notes are kept with the ignored tools, outside published docs.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "sxr_reprocessing.md"]
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence"]

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
# Only the public module is imported. Mock services used inside its methods;
# building docs needs neither the ML stack nor network access to solar data.
# NumPy stays real because the dataclasses evaluate numpy.ndarray annotations.
autodoc_mock_imports = ["astropy", "drms", "pandas", "sunpy", "tqdm"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

html_theme = "furo"
html_title = f"FOXES {release} documentation"
html_short_title = "FOXES"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["foxes.css"]
html_favicon = "_static/foxes-logo.png"
html_theme_options = {
    "source_repository": "https://github.com/griffin-goodwin/FOXES/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#963b11",
        "color-brand-content": "#963b11",
        "color-brand-visited": "#774730",
        "color-foreground-primary": "#252832",
        "color-foreground-secondary": "#565d6b",
        "color-background-secondary": "#f6f7f9",
        "color-background-border": "#e0e3e9",
        "color-api-background": "#faf6f0",
        "color-api-background-hover": "#f5eee3",
        "color-api-name": "#963b11",
        "color-api-pre-name": "#963b11",
        "color-highlight-on-target": "#fff0d4",
        "font-stack--headings": '"Avenir Next", "Segoe UI", sans-serif',
    },
    "dark_css_variables": {
        "color-brand-primary": "#ffb35b",
        "color-brand-content": "#ffb35b",
        "color-brand-visited": "#eac49f",
        "color-foreground-primary": "#e6e8ed",
        "color-foreground-secondary": "#b1b5c1",
        "color-background-primary": "#111116",
        "color-background-secondary": "#19191f",
        "color-background-border": "#30313b",
        "color-api-background": "#202026",
        "color-api-background-hover": "#282831",
        "color-api-name": "#ffc184",
        "color-api-pre-name": "#ffc184",
        "color-highlight-on-target": "#49331d",
    },
}


def linkcode_resolve(domain, info):
    """Link API objects to the source revision used by the documentation build."""
    if domain != "py" or not info.get("module"):
        return None
    obj = sys.modules.get(info["module"])
    try:
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        filename = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT)
        lines, start = inspect.getsourcelines(obj)
    except (AttributeError, TypeError, OSError, ValueError):
        return None
    revision = os.environ.get("GITHUB_SHA", "main")
    return (
        f"https://github.com/griffin-goodwin/FOXES/blob/{revision}/"
        f"{filename.as_posix()}#L{start}-L{start + len(lines) - 1}"
    )
