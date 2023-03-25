# -*- coding: utf-8 -*-
import os

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext

######################
# beginning of setup
######################


here = os.path.dirname(__file__)
if here == "":
    here = "."
packages = find_packages(where=here)
package_dir = {k: os.path.join(here, k.replace(".", "/")) for k in packages}
package_data = {}

try:
    with open(os.path.join(here, "requirements.txt"), "r") as f:
        requirements = f.read().strip(" \n\r\t").split("\n")
except FileNotFoundError:
    requirements = []
if len(requirements) == 0 or requirements == [""]:
    requirements = ["numpy", "scipy", "onnx"]

try:
    with open(os.path.join(here, "README.rst"), "r", encoding="utf-8") as f:
        long_description = "onnx-array-api:" + f.read().split("onnx-array-api:")[1]
except FileNotFoundError:
    long_description = ""

version_str = "0.1.0"
with open(os.path.join(here, "onnx_array_api/__init__.py"), "r") as f:
    line = [
        _
        for _ in [_.strip("\r\n ") for _ in f.readlines()]
        if _.startswith("__version__")
    ]
    if len(line) > 0:
        version_str = line[0].split("=")[1].strip('" ')

########################################
# C++
########################################


class cmake_build_ext(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        for ext in self.extensions:

            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = "Debug" if options["--debug"] == "ON" else "Release"

            cmake_args = [
                "-DCMAKE_BUILD_TYPE=%s" % cfg,
                # Ask CMake to place the resulting library in the directory
                # containing the extension
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                # Other intermediate static libraries are placed in a
                # temporary build directory instead
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), self.build_temp
                ),
                # Hint CMake to use the same Python executable that
                # is launching the build, prevents possible mismatching if
                # multiple versions of Python are installed
                "-DPYTHON_EXECUTABLE={}".format(sys.executable),
                # Add other project-specific CMake arguments if needed
                # ...
            ]

            # We can handle some platform-specific settings at our discretion
            if platform.system() == "Windows":
                plat = "x64" if platform.architecture()[0] == "64bit" else "Win32"
                cmake_args += [
                    # These options are likely to be needed under Windows
                    "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE",
                    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(
                        cfg.upper(), extdir
                    ),
                ]
                # Assuming that Visual Studio and MinGW are supported compilers
                if self.compiler.compiler_type == "msvc":
                    cmake_args += [
                        "-DCMAKE_GENERATOR_PLATFORM=%s" % plat,
                    ]
                else:
                    cmake_args += [
                        "-G",
                        "MinGW Makefiles",
                    ]

            cmake_args += cmake_cmd_args

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config
            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp
            )

            # Build
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", cfg], cwd=self.build_temp
            )


setup(
    name="onnx-extended",
    version=version_str,
    description="More operators for onnx reference implementation",
    long_description=long_description,
    author="Xavier Dupré",
    author_email="xavier.dupre@gmail.com",
    url="https://github.com/sdpython/onnx-extended",
    packages=packages,
    package_dir=package_dir,
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
    cmdclass={"build_ext": my_build_ext},
)
