# Usage:

## Step 1: Create build dir, and go into this dir

```sh
mkdir build
cd build

```


## Step 2: Execute the following commands to build in debug mode or release mode:

Debug mode:

```sh

cmake -DCMAKE_BUILD_TYPE=Debug ../

```

Release mode:

```sh

cmake -DCMAKE_BUILD_TYPE=Release ../


```

## Step 3: Execute make command to build this library:

```sh
make
```