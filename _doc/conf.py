import os
import sys

from onnx_array_api import __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
    "pyquickhelper.sphinxext.sphinx_epkg_extension",
    "pyquickhelper.sphinxext.sphinx_gdot_extension",
    "pyquickhelper.sphinxext.sphinx_runpython_extension",
]

templates_path = ["_templates"]
html_logo = "_static/logo.png"
source_suffix = ".rst"
master_doc = "index"
project = "onnx-array-api"
copyright = "2023, Xavier Dupré"
author = "Xavier Dupré"
version = __version__
release = __version__
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = True

html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_static_path = ["_static"]


intersphinx_mapping = {
    "onnx": ("https://onnx.ai/onnx/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": os.path.join(os.path.dirname(__file__), "examples"),
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
}

epkg_dictionary = {
    "DOT": "https://graphviz.org/doc/info/lang.html",
    "JIT": "https://en.wikipedia.org/wiki/Just-in-time_compilation",
    "onnx": "https://onnx.ai/onnx/",
    "ONNX": "https://onnx.ai/",
    "onnxruntime": "https://onnxruntime.ai/",
    "numpy": "https://numpy.org/",
    "numba": "https://numba.pydata.org/",
    "onnx-array-api": (
        "http://www.xavierdupre.fr/app/" "onnx-array-api/helpsphinx/index.html"
    ),
    "pyinstrument": "https://github.com/joerick/pyinstrument",
    "python": "https://www.python.org/",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "scipy": "https://scipy.org/",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "torch": "https://pytorch.org/docs/stable/torch.html",
}
