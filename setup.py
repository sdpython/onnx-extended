# -*- coding: utf-8 -*-
import distutils
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

try:
    import numpy
except ImportError as e:
    raise ImportError(
        f"Numpy is not installed, python _executable=f{sys.executable}."
    ) from e

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

######################
# beginning of setup
######################


here = os.path.dirname(__file__)
if here == "":
    here = "."
package_data = {
    "onnx_extended.reference.c_ops": ["*.h", "*.cpp"],
    "onnx_extended.validation": ["*.h", "*.cpp"],
}


try:
    with open(os.path.join(here, "requirements.txt"), "r") as f:
        requirements = f.read().strip(" \n\r\t").split("\n")
except FileNotFoundError:
    requirements = []
if len(requirements) == 0 or requirements == [""]:
    requirements = ["numpy", "scipy", "onnx"]

try:
    with open(os.path.join(here, "README.rst"), "r", encoding="utf-8") as f:
        long_description = "onnx-extended:" + f.read().split("onnx-extended:")[1]
except FileNotFoundError:
    long_description = ""

version_str = "0.1.0"
with open(os.path.join(here, "onnx_extended/__init__.py"), "r") as f:
    line = [
        _
        for _ in [_.strip("\r\n ") for _ in f.readlines()]
        if _.startswith("__version__")
    ]
    if len(line) > 0:
        version_str = line[0].split("=")[1].strip('" ')

########################################
# C++ Helper
########################################


def is_windows():
    return platform.system() == "Windows"


def is_darwin():
    return platform.system() == "Darwin"


def _run_subprocess(
    args,
    cwd=None,
    capture_output=False,
    dll_path=None,
    shell=False,
    env=None,
    python_path=None,
    cuda_home=None,
):
    if env is None:
        env = {}
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            if "PATH" in my_env:
                my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
            else:
                my_env["PATH"] = dll_path
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path
    # Add nvcc's folder to PATH env so that our cmake file can find nvcc
    if cuda_home:
        my_env["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + my_env["PATH"]

    if python_path:
        if "PYTHONPATH" in my_env:
            my_env["PYTHONPATH"] += os.pathsep + python_path
        else:
            my_env["PYTHONPATH"] = python_path

    my_env.update(env)

    p = subprocess.Popen(
        args,
        cwd=cwd,
        shell=shell,
        env=my_env,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )
    raise_exception = False
    while True:
        output = p.stdout.readline().decode(errors="ignore")
        if output == "" and p.poll() is not None:
            break
        if output:
            out = output.rstrip()
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
            if "fatal error" in output or "CMake Error" in output:
                raise_exception = True
    rc = p.poll()
    if raise_exception:
        raise RuntimeError(
            "'fatal error:' was found in the output. The build is stopped."
        )
    return rc


########################################
# C++ CMake Extension
########################################


class CMakeExtension(Extension):
    def __init__(self, name: str, library: str = "") -> None:
        super().__init__(name, sources=[])
        self.library_file = os.fspath(Path(library).resolve())


class cmake_build_ext(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        cfg = "Release"
        iswin = is_windows()
        isdar = is_darwin()

        cmake_cmd_args = []

        path = sys.executable
        vers = (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
        module_ext = distutils.sysconfig.get_config_var('EXT_SUFFIX')
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={path}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPYTHON_VERSION={vers}",
            f"-DPYTHON_MODULE_EXTENSION={module_ext}",
        ]
        if iswin:
            include_dir = sysconfig.get_path("include").replace("\\", "/")
            # lib_dir = sysconfig.get_config_var("LIBDIR") or ""
            # lib_dir = lib_dir.replace("\\", "/")
            numpy_include_dir = numpy.get_include().replace("\\", "/")
            cmake_args.extend(
                [
                    f"-DPYTHON_INCLUDE_DIR={include_dir}",
                    # f"-DPYTHON_LIBRARIES={lib_dir}",
                    f"-DPYTHON_NUMPY_INCLUDE_DIR={numpy_include_dir}",
                ]
            )

        cmake_args += cmake_cmd_args

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Builds the project.
        this_dir = os.path.dirname(os.path.abspath(__file__))
        build_path = os.path.abspath(self.build_temp)
        with open(
            os.path.join(os.path.dirname(__file__), ".build_path.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(build_path)
        # build_path = os.path.join(this_dir, "build")
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        source_path = os.path.join(this_dir, "cmake")

        cmd = ["cmake", "-S", source_path, "-B", build_path, *cmake_args]
        print(f"-- setup: version={sys.version_info!r}")
        print(f"-- setup: cwd={os.getcwd()!r}")
        print(f"-- setup: source_path={source_path!r}")
        print(f"-- setup: build_path={build_path!r}")
        print(f"-- setup: cmd={' '.join(cmd)}")
        _run_subprocess(cmd, cwd=build_path, capture_output=True)

        # then build
        print()
        cmd = ["cmake", "--build", build_path, "--config", cfg]
        print(f"-- setup: cwd={os.getcwd()!r}")
        print(f"-- setup: build_path={build_path!r}")
        print(f"-- setup: cmd={' '.join(cmd)}")
        _run_subprocess(cmd, cwd=build_path, capture_output=True)
        print("-- setup: done.")

        # final
        build_lib = self.build_lib

        for ext in self.extensions:
            full_name = ext._file_name
            name = os.path.split(full_name)[-1]
            if iswin:
                look = os.path.join(build_path, "Release", name)
            elif isdar:
                look = os.path.join(build_path, "Release", name)
            else:
                look = os.path.join(build_path, name)
            if not os.path.exists(look):
                raise FileNotFoundError(f"Unable to find {look!r}.")
            dest = os.path.join(build_lib, os.path.split(full_name)[0])
            if not os.path.exists(dest):
                os.makedirs(dest)
            if not os.path.exists(look):
                raise FileNotFoundError(f"Unable to find {look!r}.")
            if not os.path.exists(dest):
                raise FileNotFoundError(f"Unable to find folder {dest!r}.")
            print(f"-- copy {look!r} to {dest!r}")
            shutil.copy(look, dest)


if is_windows():
    ext = "pyd"
elif is_darwin():
    ext = "dylib"
else:
    ext = "so"


setup(
    name="onnx-extended",
    version=version_str,
    description="More operators for onnx reference implementation",
    long_description=long_description,
    author="Xavier Dupré",
    author_email="xavier.dupre@gmail.com",
    url="https://github.com/sdpython/onnx-extended",
    package_data=package_data,
    setup_requires=["numpy", "scipy"],
    install_requires=requirements,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    cmdclass={"build_ext": cmake_build_ext},
    ext_modules=[
        CMakeExtension(
            "onnx_extended.validation.vector_function_cy",
            f"onnx_extended/validation/vector_function_cy.{ext}",
        ),
        CMakeExtension(
            "onnx_extended.validation._validation",
            f"onnx_extended/validation/_validation.{ext}",
        ),
        CMakeExtension(
            "onnx_extended.reference.c_ops.c_op_conv_",
            f"onnx_extended/reference/c_ops/c_op_conv_.{ext}",
        ),
        CMakeExtension(
            "onnx_extended.reference.c_ops.c_op_tree_ensemble_py_",
            f"onnx_extended/reference/c_ops/c_op_tree_ensemble_py_.{ext}",
        ),
    ],
)
