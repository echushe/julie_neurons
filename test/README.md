# Usage:

## Step 1: Create build dir, and go into this dir

```sh
mkdir build
cd build

```


## Step 2: Execute the following commands to build:

- Debug mode :

```sh
cmake -DCMAKE_BUILD_TYPE=Debug ../

```

- Release mode:

```sh
cmake -DCMAKE_BUILD_TYPE=Release ../

```

- If you want to add OpenBLAS dependency to speed up calculations on CPU, you can turn OpenBLAS on.

```sh
cmake -DWITH_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Debug ../
```

or

```sh
cmake -DWITH_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Release ../
```

- If you want to add OneDNN dependency to continue to boost deep learning on Intel CPU, you can turn OneDNN (MKL) on.

```sh
cmake -DWITH_OPENBLAS=ON -DWITH_ONEDNN=ON -DCMAKE_BUILD_TYPE=Debug ../
```

or

```sh
cmake -DWITH_OPENBLAS=ON -DWITH_ONEDNN=ON -DCMAKE_BUILD_TYPE=Release ../
```

- If you want to speed up calculations with your NVIDIA GPU card, you should add an extra option named **WITH_CUDA** and turn it on.

```sh
cmake -DWITH_CUDA=ON -DCMAKE_BUILD_TYPE=Debug ../
```

or

```sh
cmake -DWITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release ../
```

- If you want to continue to boost deep learning with NVIDIA GPU card, you should turn cuDNN on.

```sh
cmake -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_BUILD_TYPE=Debug ../
```

or

```sh
cmake -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_BUILD_TYPE=Release ../
```

## Step 3: Execute make command to build this library:

```sh
make
```
You can speed up the build if you specify number of jobs to run.
For example:

```
make -j8
```
