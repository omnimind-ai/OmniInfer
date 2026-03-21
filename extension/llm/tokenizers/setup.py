# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# @lint-ignore-every LICENSELINT
# type: ignore[syntax]

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py as build_py_orig

# Read the README file
with open("README.md", "r") as f:
    long_description = f.read()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):  # noqa C901
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            "-DSUPPORT_REGEX_LOOKAHEAD=ON",
            "-DTOKENIZERS_BUILD_PYTHON=ON",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        ]
        build_args = ["--target", "pytorch_tokenizers_cpp"]
        build_tool_args: list[str] = []

        # Use Clang for Windows builds.
        if sys.platform == "win32":
            cmake_args += ["-T ClangCL"]

        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSX on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja  # noqa: F401

                    ninja_executable_path = os.path.join(ninja.BIN_DIR, "ninja")
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            build_tool_args = ["--", "/p:TrackFileAccess=false"]
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", "x64"]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", "."] + build_args + build_tool_args,
            cwd=build_temp,
            check=True,
        )


class BuildPy(build_py_orig):
    """Ensure header files are copied into the package during build."""

    def run(self):
        super().run()
        headers_src = Path("include")
        if not headers_src.exists():
            return

        headers_dst = Path(self.build_lib) / "pytorch_tokenizers" / "include"
        for file_path in headers_src.rglob("*"):
            if file_path.is_file():
                destination = headers_dst / file_path.relative_to(headers_src)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, destination)


setup(
    name="pytorch-tokenizers",
    version="1.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meta-pytorch/tokenizers",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "pytorch_tokenizers": [
            "include/*.h",
            "include/**/*.h",
            "include/*.hpp",
            "include/**/*.hpp",
        ]
    },
    ext_modules=[CMakeExtension("pytorch_tokenizers.pytorch_tokenizers_cpp")],
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": BuildPy,
    },
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=[
        "pybind11>=2.6.0",
        "sentencepiece",
        "mistral-common",
        "tokenizers",
        "tiktoken",
    ],
    setup_requires=[
        "pybind11>=2.6.0",
        "cmake>=3.18",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
