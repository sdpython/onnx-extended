Install CUDA on WSL (2)
=======================

Distribution
++++++++++++

The installation was done with distribution Ubuntu 22.04.
Let's assume WSL is installed, otherwise, here are some useful commands.

::

    # see all local distributions
    wsl -s -l

    # see available distributions online
    wsl --list --online

    # install one distribution or download it
    wsl --install -d Ubuntu-22.04

CUDA
++++

The CUDA driver must be installed as well. It can be downloaded from
`NVIDIA Driver Downloads <https://www.nvidia.com/download/index.aspx>`_.
Make sure you are using the one from your graphics card.
Installation of required packages. The following instruction were tested
with CUDA==11.8.

::

    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt autoremove -y
    sudo apt-get install -y cmake zlib1g-dev libssl-dev python3-dev libhwloc-dev libevent-dev libcurl4-openssl-dev libopenmpi-dev clang unzip

Let's install :epkg:`gcc`:

::

    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt autoremove -y
    sudo apt install -y libcurl4 ca-certificates
    sudo apt-get install -y gcc g++
    gcc --version

Installation of `ninja <https://github.com/ninja-build/ninja/>`_:

::

    wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip
    unzip ninja-linux.zip
    sudo cp ./ninja /usr/local/bin/
    sudo chmod a+x /usr/local/bin/ninja

Installation of :epkg:`cmake`.

::

    mkdir install
    cd install
    export cmake_version=3.26.4
    curl -OL https://github.com/Kitware/CMake/releases/download/v${cmake_version}/cmake-${cmake_version}.tar.gz
    tar -zxvf cmake-${cmake_version}.tar.gz
    cd cmake-${cmake_version}
    ./bootstrap --system-curl
    make
    sudo make install
    # This line should not be necessary.
    # export PATH=~/install/cmake-${cmake_version}/bin/:$PATH
    cmake --version

Installation of CUDA (choose a compatible version with :epkg:`pytorch`, 11.8 for example).

See `CUDA on WSL User Guide
<https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2>`_

::

    export CUDA_VERSION=11.8
    export CUDA_VERSION_=11-8
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}.0/local_installers/cuda-repo-wsl-ubuntu-${CUDA_VERSION_}-local_${CUDA_VERSION}.0-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-${CUDA_VERSION_}-local_${CUDA_VERSION}.0-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-9EA88183-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda nvidia-cuda-toolkit

Now you may run `nvidia-smi -L` to list the available GPUs.

Installation of :epkg:`cudnn` (after it is downloaded):

::

    sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.2.26_1.0-1_amd64.deb
    sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.2.26/cudnn-local-D7CBF0C2-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get install libcudnn8  libcudnn8-dev

Installation of :epkg:`nccl` (after it is downloaded):

See `Install NCCL <https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html>`_.

::

    cd /usr/local
    sudo tar xvf nccl_2.18.1-1+cuda11.0_x86_64.txz
    # sudo apt install libnccl2 libnccl-dev
    export PATH=/usr/local/nccl_2.18.1-1+cuda11.0_x86_64/:$PATH

Installation of pip and update python packages:

::

    sudo apt-get install -y python3-pybind11 libpython3.10-dev
    wget https://bootstrap.pypa.io/get-pip.py
    sudo python3 get-pip.py
    sudo python3 -m pip install --upgrade numpy jupyter pandas statsmodels scipy scikit-learn pybind11 cython flatbuffers mpi4py notebook nbconvert flatbuffers pylint autopep8 sphinx sphinx-gallery cffi black py-spy fire pytest

Torch
+++++

Installation of :epkg:`pytorch` of it is available for CUDA 11.8:

::

    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Otherwise, it has to be built from sources:

::

    wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
    bash Anaconda3-2022.10-Linux-x86_64.sh
    conda create -p ~/install/acond10 python=3.10
    conda activate ~/install/acond10
    conda install -y astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
    conda install -y mkl mkl-include
    conda install -c pytorch magma-cuda118
    mkdir ~/github
    cd ~/github
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    # python tools/amd_build/build_amd.py
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    export CUDA_VERSION=11.8
    export CUDACXX=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
    export USE_ITT=0
    export USE_KINETO=0
    export BUILD_TEST=0
    export USE_MPI=0
    export BUILD_CAFFE2=0
    export BUILD_CAFFE2_OPS=0
    export USE_DISTRIBUTED=0
    export MAX_JOBS=2
    python setup.py build

Then to check CUDA is available:

::

    import torch
    print(torch.cuda.is_available())

onnxruntime-training
++++++++++++++++++++

Build :epkg:`onnxruntime-training` before :epkg:`onnx`
to build :epkg:`protobuf` as well.

::

    alias python=python3
    export CUDA_VERSION=11.8
    export CUDACXX=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
    export CMAKE_CUDA_COMPILER=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc
    python3 ./tools/ci_build/build.py --build_dir ./build/linux_cuda --config Release --build_shared_lib --enable_training --use_cuda --cuda_version=${CUDA_VERSION} --cuda_home /usr/local/cuda-${CUDA_VERSION}/ --cudnn_home /usr/local/cuda-${CUDA_VERSION}/ --build_wheel --parallel 2 --skip_test

Option ``--parallel 1`` can be used to fix the parallelism while building onnxruntime.

Another option is to use a docker:
`Running Existing GPU Accelerated Containers on WSL 2
<https://docs.nvidia.com/cuda/wsl-user-guide/index.html#ch05-running-containers>`_.

onnx
++++

Then onnx built inplace:

::

    git clone https://github.com/onnx/onnx.git
    cd onnx
    python -m pip install -e .

PYTHONPATH
++++++++++

Some useful commands:

::

    export PYTHONPATH=~/github/onnx:~/github/onnxruntime/build/linux_cuda/Release/Release
    export PYTHONPATH=$PYTHONPATH:~/github/onnx-extended:~/github/onnx-array-api