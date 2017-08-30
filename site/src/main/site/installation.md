---
layout: page
title:  "Installation"
section: "installation"
position: 2
---

# Installation

You first need to make sure you have the TensorFlow dynamic library 
installed. You can either download pre-compiled versions of it, or you 
can compile it yourself from the TensorFlow sources.

**NOTE:** *The pre-compiled versions that you can download are not 
currently supported. Please compile the native library yourself. The 
pre-compiled versions will start working again once their updated as 
the Scala API depends on some functionality that was added after the 
last release.*

## Downloading the TensorFlow Dynamic Library

You can download it from one of the following links:
  - **Linux:**
    - CPU-only: [https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.3.0.tar.gz](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.3.0.tar.gz)
    - GPU-enabled: [https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.3.0.tar.gz](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.3.0.tar.gz)
  - **Mac:**
    - CPU-only: [https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.3.0.tar.gz](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.3.0.tar.gz)
  - **Windows:** 
    - CPU-only: [https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.3.0.zip](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.3.0.zip)

## Compiling the TensorFlow Dynamic Library

Perform the following steps:
  1. Clone the 
     [TensorFlow repository](https://github.com/tensorflow/tensorflow) 
     in a directory (e.g., `tensorflow`).
  2. Compile the TensorFlow dynamic library by running the following
     commands in that directory:

     ```bash
     ./configure
     bazel build --config=opt //tensorflow:libtensorflow.so
     ```

     Make sure to add the `--config=cuda` option when running the last
     command, if compiling with CUDA support.
  3. Copy the `bazel-bin/tensorflow/libtensorflow.so` (possibly having 
     a different extension, depending on the platform you're using) 
     file in a directory that is in `LD_LIBRARY_PATH`, or set 
     `LD_LIBRARY_PATH` appropriately.
     
## Installing the Protocol Buffers Compiler

You also need protoc, the Protocol Buffers compiler.

On Debian/Ubuntu, you can install it with APT:

```bash
apt install protobuf-compiler
```

You can also download prebuild binaries from [https://github.com/google/protobuf/releases/](https://github.com/google/protobuf/releases/)
(choose the protoc variant appropriate for your platform).

## Compiling the TensorFlow Scala API

First make sure you have [CMake](https://cmake.org/install/) installed, 
and that the TensorFlow dynamic library is in the path. Then, you can  
compile the Scala API by running the following command from within the 
`tensorflow_scala` directory:

```bash
sbt compile
```
