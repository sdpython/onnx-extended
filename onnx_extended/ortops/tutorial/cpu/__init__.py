import os
import platform
from typing import List

_ort_ext_libs = []


def get_ort_ext_libs() -> List[str]:
    """
    Returns the list of libraries implementing new simple
    :epkg:`onnxruntime` kernels implemented for the
    :epkg:`CPUExecutionProvider`.
    """
    global _ort_ext_libs
    if len(_ort_ext_libs) == 0:
        if platform.system() == "Windows":
            ext = ".dll"
        elif platform.system() == "Darwin":
            ext = ".dylib"
        else:
            ext = ".so"
        this = os.path.abspath(os.path.dirname(__file__))
        files = os.listdir(this)
        res = []
        for name in files:
            e = os.path.splitext(name)[-1]
            if e == ext and "ortops" in name:
                res.append(os.path.join(this, name))
        if len(res) == 0:
            raise RuntimeError(
                f"Unable to find any kernel library with ext={ext!r} "
                f"in {this!r} among {files}."
            )
        _ort_ext_libs = res
    return _ort_ext_libs
