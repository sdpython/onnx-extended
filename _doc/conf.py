import os
import sys
from onnx_extended import __version__

for name in ["CHANGELOGS.rst", "LICENSE.txt"]:
    s = os.path.join(os.path.dirname(__file__), "..", name)
    if name == "LICENSE.txt":
        d = os.path.join(os.path.dirname(__file__), name.replace(".txt", ".rst"))
        with open(s, "r", encoding="utf-8") as f:
            with open(d, "w", encoding="utf-8") as g:
                g.write("LICENSE\n=======\n\n")
                g.write(f.read())
    else:
        d = os.path.join(os.path.dirname(__file__), name)
        with open(s, "r", encoding="utf-8") as f:
            with open(d, "w", encoding="utf-8") as g:
                g.write(f.read())

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
    "sphinx_issues",
    "matplotlib.sphinxext.plot_directive",
    "pyquickhelper.sphinxext.sphinx_epkg_extension",
    "pyquickhelper.sphinxext.sphinx_gdot_extension",
    "pyquickhelper.sphinxext.sphinx_runpython_extension",
]

templates_path = ["_templates"]
html_logo = "_static/logo.png"
source_suffix = ".rst"
master_doc = "index"
project = "onnx-extended"
copyright = "2023, Xavier Dupré"
author = "Xavier Dupré"
version = __version__
release = __version__
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = True
issues_github_path = "sdpython/onnx-extended"

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
    "cmake": "https://cmake.org/",
    "CPUExecutionProvider": "https://onnxruntime.ai/docs/execution-providers/",
    "cublasLtMatmul": "https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasltmatmul#cublasltmatmul",
    "CUDA": "https://developer.nvidia.com/",
    "cudnn": "https://developer.nvidia.com/cudnn",
    "cython": "https://cython.org/",
    "DOT": "https://graphviz.org/doc/info/lang.html",
    "eigen": "https://eigen.tuxfamily.org/",
    "gcc": "https://gcc.gnu.org/",
    "JIT": "https://en.wikipedia.org/wiki/Just-in-time_compilation",
    "nccl": "https://developer.nvidia.com/nccl",
    "numpy": "https://numpy.org/",
    "numba": "https://numba.pydata.org/",
    "nvidia-smi": "https://developer.nvidia.com/nvidia-system-management-interface",
    "nvprof": "https://docs.nvidia.com/cuda/profiler-users-guide/index.html",
    "onnx": "https://onnx.ai/onnx/",
    "ONNX": "https://onnx.ai/",
    "onnxruntime": "https://onnxruntime.ai/",
    "onnxruntime-training": "https://github.com/microsoft/onnxruntime/tree/master/orttraining",
    "onnxruntime releases": "https://github.com/microsoft/onnxruntime/releases",
    "onnx-array-api": (
        "http://www.xavierdupre.fr/app/" "onnx-array-api/helpsphinx/index.html"
    ),
    "onnxruntime C API": "https://onnxruntime.ai/docs/api/c/",
    "onnxruntime Graph Optimizations": (
        "https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html"
    ),
    "openmp": "https://www.openmp.org/",
    "protobuf": "https://github.com/protocolbuffers/protobuf",
    "pybind11": "https://github.com/pybind/pybind11",
    "pyinstrument": "https://github.com/joerick/pyinstrument",
    "python": "https://www.python.org/",
    "pytorch": "https://pytorch.org/",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "scipy": "https://scipy.org/",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "torch": "https://pytorch.org/docs/stable/torch.html",
    "WSL": "https://docs.microsoft.com/en-us/windows/wsl/install",
}
