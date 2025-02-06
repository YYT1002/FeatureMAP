# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'FeatureMap'
copyright = '2024, Yang Yang'
author = 'Yang Yang'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "autoapi.extension"
]

myst_enable_extensions = [
    "html_image",   # Enable HTML images
    "colon_fence"   # Enable ::: fences
]

autoapi_dirs = ['../featuremap']
autoapi_ignore = ['*/tests/*']

# conf.py
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]

autoapi_root = 'autoapi'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
import os
docs_dir = os.path.abspath(os.path.dirname(__file__))

project_root = os.path.abspath(os.path.join(docs_dir, '..'))

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_extra_path = [os.path.join(project_root, 'figures')]

# -- Options for EPUB output
epub_show_urls = 'footnote'

# In your conf.py
exclude_patterns = ['tests', '__init__.py']  # Exclude directories or files
