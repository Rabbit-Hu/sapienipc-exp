# sapienipc-exp
SapienIPC experimental release. This release is temporary and will not be maintained. We will release a stable version soon.

---

Copyright 2024                             
Xiaodi Yuan                                                
University of California San Diego                                         
yuanxiaodidl@gmail.com                                                      
                                                                         
This program may be freely redistributed under the condition that the copyright
notices (including this entire header) are not removed, and no compensation is
received. Private, research, and institutional use is free. You may distribute
modified versions of this code UNDER THE CONDITION THAT THIS CODE AND ANY
MODIFICATIONS MADE TO IT IN THE SAME FILE REMAIN UNDER COPYRIGHT OF THE ORIGINAL
AUTHOR, BOTH SOURCE AND OBJECT CODE ARE MADE FREELY AVAILABLE WITHOUT CHARGE,
AND CLEAR NOTICE IS GIVEN OF THE MODIFICATIONS. Distribution of this code as
part of a commercial system is permissible ONLY BY DIRECT ARRANGEMENT WITH THE
AUTHOR. (If you are not directly supplying this code to a customer, and you are
instead telling them how they can obtain it for free, then you are not required
to make any arrangement with me.)

---


## Requirements:

For running Warp:
- Python 3.8.x-3.11.x
- `pip install numpy`

For running SapienIPC: 
- `pip install meshio`
- PyTorch version that is compatible with your CUDA

For building Warp:
- GCC 7.2 upwards (Linux)
- CUDA Toolkit 11.8 or higher
- Git LFS installed (https://git-lfs.github.com/)

## Installation

### Step 1: Clone this repo recursively
First, clone this repo recursively so that the submodule `warp_` is cloned:

```shell
git clone --recurse-submodules git@github.com:Rabbit-Hu/sapienipc-exp.git
cd sapienipc-exp
pip install -r requirements.txt
```

### Step 2: Install our fork of Warp

Then build our fork of Warp in the `warp_` submodule:

```shell
cd warp_
python build_lib.py --cuda_path /usr/local/cuda  # Replace with your cuda path 
```

Install the Warp package into your python environment:

```shell
pip install -e .
```

### Step 2: Install SAPIEN

Use pip to install the latest SAPIEN 3 wheel from [SAPIEN Nightly Release](https://github.com/haosulab/SAPIEN/releases/tag/nightly). Look for your own python version. 

The installation command should look like this:

```shell
pip install https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev{SOME_DATE}-cp{PYTHON_VERSION}-cp3{PYTHON_VERSION}-manylinux2014_x86_64.whl
```

See the [SAPIEN](https://github.com/haosulab/SAPIEN) repo for reference.


### Step 3: Install Warp IPC for SAPIEN (this Repo)

```shell
cd sapienipc-exp  # directory of this repo
pip install -e .
```

You can try running an example:

```shell
python examples/example_peg.py
```
