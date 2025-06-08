# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

# Add project root to sys.path for autodoc
sys.path.insert(0, str(Path("..").resolve()))

# Project information
project = "Energy Transformer"
copyright = "2025, Ayan Das"  # noqa: A001
author = "Ayan Das"
release = "0.3.1"
version = "0.3.1"

# General configuration
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    # Additional extensions
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "numpydoc",
]

# Templates and patterns
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Language
language = "en"

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": False,
}

autosummary_generate = True
autosummary_imported_members = False

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "Tensor": "torch.Tensor",
    "Module": "torch.nn.Module",
    "Device": "torch.device",
    "Dtype": "torch.dtype",
}

# Type hints
typehints_defaults = "comma"
typehints_use_signature = True
typehints_use_signature_return = True

# Intersphinx mapping (link to other projects)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "einops": ("https://einops.rocks/", None),
}

# MyST parser for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "tasklist",
]

# HTML output with PyData theme
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/b-vitamins/energy-transformer",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/energy-transformer/",
            "icon": "fa-brands fa-python",
        },
    ],
    "logo": {
        "text": "Energy Transformer",
    },
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "show_prev_next": True,
    "use_edit_page_button": True,
    "navbar_align": "left",
    "header_links_before_dropdown": 4,
    "primary_sidebar_end": ["indices.html"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "pygment_light_style": "default",
    "pygment_dark_style": "monokai",
}

html_context = {
    "github_user": "b-vitamins",
    "github_repo": "energy-transformer",
    "github_version": "main",
    "doc_path": "docs",
}

# Custom CSS (create if needed)
html_css_files = [
    "custom.css",
]

# Copy button configuration
copybutton_prompt_text = (
    r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True

# LaTeX/PDF output
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": r"""
\usepackage{amsmath,amssymb}
\setcounter{tocdepth}{2}
""",
}

# Suppress specific warnings
suppress_warnings = ["autodoc.import_object"]
