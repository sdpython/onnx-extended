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
known_extensions = [
    "*.cc",
    "*.cpp",
    "*.cu",
    "*.cuh",
    "*.dylib",
    "*.h",
    "*.hpp",
    "*.pyd",
    "*.so",
]
package_data = {
    "onnx_extended.ortcy.wrap": known_extensions,
    "onnx_extended.reference.c_ops.cpu": known_extensions,
    "onnx_extended.validation.cpu": known_extensions,
    "onnx_extended.validation.cython": known_extensions,
    "onnx_extended.validation.cuda": known_extensions,
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


def find_cuda():
    try:
        p = subprocess.Popen(
            "nvidia-smi",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return False
    while True:
        output = p.stdout.readline().decode(errors="ignore")
        if output == "" and p.poll() is not None:
            break
        if output:
            if "CUDA Version:" in output:
                return True
    p.poll()
    return False


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

    if is_windows():
        py_path = os.path.dirname(sys.executable)
        if "PATH" in my_env:
            my_env["PATH"] = py_path + os.pathsep + my_env["PATH"]
        else:
            my_env["PATH"] = py_path

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
            if (
                "fatal error" in output
                or "CMake Error" in output
                or "gmake: ***" in output
                or "): error C" in output
            ):
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
    user_options = [
        *build_ext.user_options,
        ("enable-nvtx=", None, "Enables compilation with NVTX events."),
        ("with-cuda=", None, "If cuda is available, CUDA is "
                            "used by default unless this option is set to 0"),
    ]

    def initialize_options(self):
        self.enable_nvtx = None
        self.with_cuda = None
        build_ext.initialize_options(self)

    def finalize_options(self):
        b_values = {None, 0, 1, "1", "0", True, False}
        if self.enable_nvtx not in b_values:
            raise ValueError(f"enable_nvtx={self.enable_nvtx!r} must be in {b_values}.")
        if self.with_cuda not in b_values:
            raise ValueError(f"with_cuda={self.with_cuda!r} must be in {b_values}.")
        self.enable_nvtx = self.enable_nvtx in {1, "1", True, "True"}
        self.with_cuda = self.with_cuda in {1, "1", True, "True", None}
        build_ext.finalize_options(self)

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
        versmm = f"{sys.version_info.major}." f"{sys.version_info.minor}."
        module_ext = distutils.sysconfig.get_config_var("EXT_SUFFIX")
        cmake_args = [
            f"-DPYTHON_EXECUTABLE={path}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPYTHON_VERSION={vers}",
            f"-DPYTHON_VERSION_MM={versmm}",
            f"-DPYTHON_MODULE_EXTENSION={module_ext}",
        ]

        if os.environ.get("USE_NVTX", "0") in (1, "1") or self.enable_nvtx:
            cmake_args.append("-DUSE_NVTX=1")
        if os.environ.get("USE_CUDA", "1") in (0, "0") or not self.with_cuda:
            cmake_args.append("-DUSE_CUDA=0")
        else:
            cmake_args.append("-DUSE_CUDA=1")

        if iswin or isdar:
            include_dir = sysconfig.get_paths()["include"].replace("\\", "/")
            lib_dir = (
                sysconfig.get_config_var("LIBDIR")
                or sysconfig.get_paths()["stdlib"]
                or ""
            ).replace("\\", "/")
            numpy_include_dir = numpy.get_include().replace("\\", "/")
            cmake_args.extend(
                [
                    f"-DPYTHON_INCLUDE_DIR={include_dir}",
                    # f"-DPYTHON_LIBRARIES={lib_dir}",
                    f"-DPYTHON_LIBRARY_DIR={lib_dir}",
                    f"-DPYTHON_NUMPY_INCLUDE_DIR={numpy_include_dir}",
                    # "-DUSE_SETUP_PYTHON=1",
                    f"-DPYTHON_NUMPY_VERSION={numpy.__version__}",
                ]
            )
            os.environ["PYTHON_NUMPY_INCLUDE_DIR"] = numpy_include_dir

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
            else:
                look = os.path.join(build_path, name)
            if not os.path.exists(look):
                content = os.listdir(build_path)
                raise FileNotFoundError(
                    f"Unable to find {look!r}, " f"build_path contains {content}."
                )
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

cuda_extensions = []
has_cuda = find_cuda()
if has_cuda:
    add_cuda = True
    if "--with-cuda" in sys.argv:
        pos = sys.argv.index("--with-cuda")
        if len(sys.argv) > pos + 1 and sys.argv[pos+1] in ("0", 0, False, "False"):
            add_cuda = False
    if add_cuda:
        cuda_extensions.extend(
            [
                CMakeExtension(
                    "onnx_extended.validation.cuda.cuda_example_py",
                    f"onnx_extended/validation/cuda/cuda_example_py.{ext}",
                )
            ]
        )

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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    cmdclass={"build_ext": cmake_build_ext},
    ext_modules=[
        CMakeExtension(
            "onnx_extended.validation.cython.vector_function_cy",
            f"onnx_extended/validation/cython/vector_function_cy.{ext}",
        ),
        CMakeExtension(
            "onnx_extended.validation.cpu._validation",
            f"onnx_extended/validation/cpu/_validation.{ext}",
        ),
        CMakeExtension(
            "onnx_extended.reference.c_ops.cpu.c_op_conv_",
            f"onnx_extended/reference/c_ops/cpu/c_op_conv_.{ext}",
        ),
        CMakeExtension(
            "onnx_extended.reference.c_ops.cpu.c_op_tree_ensemble_py_",
            f"onnx_extended/reference/c_ops/cpu/c_op_tree_ensemble_py_.{ext}",
        ),
        *cuda_extensions,
    ],
)
