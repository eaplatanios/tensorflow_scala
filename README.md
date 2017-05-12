# TensorFlow Scala

[![Build Status](https://travis-ci.com/eaplatanios/tensorflow_scala.svg?token=VBPxqvcGXTuwbjkVyN68&branch=master)](https://travis-ci.com/eaplatanios/tensorflow_scala)

TensorFlow API for the Scala Programming Language

## Set Up for Development

Make sure you have [CMake](https://cmake.org/install/) installed and
then perform the following steps:
  1. Clone this repository in a directory (e.g., `tensorflow_scala`).
  2. Compile the TensorFlow dynamic library by running the following
     commands in the
     [TensorFlow](https://github.com/tensorflow/tensorflow) source code
     directory:

     ```bash
     ./configure
     bazel build --config=opt //tensorflow:libtensorflow.so
     ```

     Make sure to add the `--config=cuda` option when running the last
     command, if compiling with CUDA support.
  3. Copy the `bazel-bin/tensorflow/libtensorflow.so` file in a
     directory that is in `LD_LIBRARY_PATH`, or set `LD_LIBRARY_PATH`
     appropriately.

In order to compile, run the following command from within the
`tensorflow_scala` directory:

```bash
sbt compile
```

## Supported Features

- General features:
  - [x] Slice creation
  - [x] Op output slicing
  - [ ] Variables slicing
  - [ ] Slice assignment
- Op creation API helpers:
  - [x] Graph op creation context
  - [x] Name scope op creation context
  - [x] Device op creation context
  - [x] Colocation op creation context
  - [x] Control dependencies op creation context
  - [x] Attribute op creation context
  - [x] Container op creation context
  - [x] Graph collections
  - [x] Variables support
  - [ ] Control flow ops support
  - [x] Gradients support
  - [ ] Gradient override map op creation context (need gradient support first)
  - [ ] Kernel label map op creation context (may be unnecessary)
- Execution API helpers:
  - [ ] Default session
  - [ ] Session execution context
  - [ ] Session reset functionality
- Tensor API:
  - [ ] More flexible/efficient slicing for obtaining and assigning elements
  - [ ] More numpy-like operations for tensors
- General API features:
  - [x] Support for all data types.
  - [ ] Summaries
  - [x] Optimizers
  - [ ] Estimators
  - [ ] tfprof / op statistics collection

## TODOs

- Switch to using "Seq" instead of "Array" wherever possible
- Tensors:
  - Overloaded unary, binary, and comparison operators (data type aware)
  - Convenient string conversion methods
  - More efficient slicing (specialized contiguous slicing)
- Op creation:
  - Add tests for all of the op functions
  - Get graph from inputs
  - Assert same graph for inputs and control inputs
  - Convert to tensor function (use implicit conversions?)
  - Support re-entering existing name scopes
  - Reset default graph
  - Register op statistics
- Execution:
  - Revamp the session API
    - Multiple overloaded versions of "run" covering all of the most common use cases
