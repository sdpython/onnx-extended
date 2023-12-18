import os
import platform
from typing import Dict, List

_ort_ext_libs_pathes: Dict[str, List[str]] = {}


def _get_ort_ext_libs(path: str) -> List[str]:
    """
    Returns the list of libraries implementing new simple
    :epkg:`onnxruntime` kernels and places in folder *path*.
    """
    global _ort_ext_libs_pathes
    if path not in _ort_ext_libs_pathes:
        _ort_ext_libs_pathes[path] = []
    if not _ort_ext_libs_pathes[path]:
        if platform.system() == "Windows":
            ext = ".dll"
        elif platform.system() == "Darwin":
            ext = ".dylib"
        else:
            ext = ".so"
        this = os.path.abspath(path)
        files = os.listdir(this)
        res = []
        for name in files:
            e = os.path.splitext(name)[-1]
            if e == ext and "ortops" in name:
                res.append(os.path.join(this, name))
        assert res, (
            f"Unable to find any kernel library with ext={ext!r} "
            f"in {this!r} among {files}."
        )
        _ort_ext_libs_pathes[path] = res
    return _ort_ext_libs_pathes[path]
