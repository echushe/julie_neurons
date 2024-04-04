# Julie Neurons (JN)
> An add-on for deep learning framework DeepFrame, CUDA and OpenCL. 

## This repository includes a library of deep learning framework, a unit test program of for this library, and some demo programs of neural networks depending on this library. They are all implemented by C++.
## With JN, a quick graph construction mechanism is introduced so that more complex deep networks can be created more easily.
## This reporsitory is collaborated with Ning Xu (xuningandy@outlook.com).

Apology for the possible delay due to pandemics, we are working to our full extent. This extra framework is derived from another repository named DeepFrame. 

## There are 4 versions of JN ##
* The library supporting CPU by default
* The library of CUDA version that can speed up neural calculations on NVIDIA GPU specifically.
* The library of oneDNN (MKL) version that can speed up neural calculations on Intel CPU & GPU (Partly developed) 
* The library of OpenCL version that can support hardware plarforms of even broader range (It is still under development).

## There are several components in this repository: ##

* The **julie_neurons** directory which includes the framework itself
* The **OpenBLAS** directory where OpenBLAS is installed. This directory is empty by default. You can download and compile OpenBLAS source code from https://github.com/xianyi/OpenBLAS and install it here.
* The **oneDNN** directory where oneDNN (MKL) is installed. Like OpenBLAS, this directory is empty by default. You can download and compile oneDNN source code from https://github.com/oneapi-src/oneDNN and install it here.
* The **test** directory which includes unit test cases
* The **demo** directory including some demo programs of neural networks
* All **demo** programs use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. You have to download the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in advance.

### You can go to each diretory to build the library and demos respectively by following README manuals. ##

## There are some optional dependencies for Julie Neurons that need extra installation procedures of third party components: openBLAS, oneDNN, CUDA and cuDNN.
## I. OpenBLAS dependency (optional) ##

- ### You should install OpenBLAS in advance if you want to build Julie Neurons with flag **`WITH_OPENBLAS`** turned on in CMake.
### Step 1: Download OpenBLAS source code from https://github.com/xianyi/OpenBLAS

```sh
git clone https://github.com/xianyi/OpenBLAS.git
```

### Step 2: Compile OpenBLAS

```sh
cd OpenBLAS
make
```

### Step 3: Install OpenBLAS into `OpenBLAS` directory inside julie_neurons reporsitory directory alongside with julie_neurons library directory

```
make install PREFIX=location_of_julie_neurons_repository/OpenBLAS
```

## II. oneDNN (MKL) dependency (optional) ##

- ### oneDNN should get installed in advance if you want to build Julie Neurons with flag **`WITH_ONEDNN`** turned on in CMake.

- ### oneDNN can only get installed on x86 platform only.

### Step 1: Download oneDNN source code from https://github.com/oneapi-src/oneDNN

```sh
git clone https://github.com/oneapi-src/oneDNN.git
```

### Step 2: Go to oneDNN documentation https://oneapi-src.github.io/oneDNN/dev_guide_build.html and follow compilation and building steps that match your hardware & system configurations. A new directory named `build` will be created.

### Step 3: Copy `include` directory into `location_of_julie_neurons_repository/oneDNN`

### Step 4: Create a directory named `lib` in `location_of_julie_neurons_repository/oneDNN/lib`, and copy all files with names beginning with `lib` from `build/src/` into `location_of_julie_neurons_repository/oneDNN/lib`

## III. CUDA dependency (optional)

- ### CUDA dependency requires NVIDIA GPU of at least 5.3 computing capacity.

- ### CUDA of at least version 8.0 should get installed before you build Julie Neurons with flag **`WITH_CUDA`** turned on in CMake.

- ### Please refer to NVIDIA's official page of CUDA driver and CUDA toolkit for more details of installation of CUDA.

## IV. cuDNN dependency (optional)

- ### cuDNN should get installed in advance if you would like to boost Julie Neurons with cuDNN. Please read NVIDIA's official document of cuDNN installation guide here carefully: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
- ### Turn flag **`WITH_CUDNN`** on in CMake if you want to build Julie Neurons with cuDNN dependency.



