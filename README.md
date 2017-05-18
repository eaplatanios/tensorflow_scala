# TensorFlow API for Scala

[![Build Status](https://travis-ci.com/eaplatanios/tensorflow_scala.svg?token=VBPxqvcGXTuwbjkVyN68&branch=master)](https://travis-ci.com/eaplatanios/tensorflow_scala)

This is an attempt to replicate most of the TensorFlow Python API 
functionality in Scala. It is a work in progress and a project I started 
working on for my personal research purposes. The API is thus very 
likely to keep changing until I settle with something that makes me 
happy. Having said that, I try to stick as close as possible to the 
Python API usage to make transition easy for users and much of what I 
have already implemented is likely to stay as is.

## Main Features

- Exposed API namespace similar to that of the Python API. For example:
  - `tf.constant(...)` creates a constant op.
  - `tf.reshape(...)` creates a reshape op.
  - `tf.Graph(...)` creates a graph.
  - `tf.Session(...)` creates a session.
  - etc.
- Straightforward API for graph creation. For example:
  ```scala
  val inputs = tf.placeholder(tf.FLOAT32, tf.Shape(-1, 10))
  val outputs = tf.placeholder(tf.FLOAT32, tf.Shape(-1, 10))
  val predictions = tf.createWith(nameScope = "Linear") {
    val weights = tf.variable("weights", tf.Shape(10, 1), tf.FLOAT32, tf.zerosInitializer)
    val predictions = tf.matMul(inputs, weights)
    predictions
  }
  val loss = tf.sum(tf.square(predictions - outputs))
  val optimizer = tf.train.AdaGrad(1.0)
  val trainOp = optimizer.minimize(loss)
  ```
- Efficient interaction with the native library that avoids unnecessary 
  copying of data.
  - All tensors are backed by a `DirectByteBuffer` so that the memory can 
    be shared with the native TensorFlow library.
  - For tensors that are created in the Scala API and passed on the 
    native library (e.g., fed into a TensorFlow session), we create a 
    global reference to make the JVM garbage collector aware of the fact 
    that the native library is using that 
    tensor. We also pass a custom deallocator to the native library that 
    simply deletes that global reference.
  - For tensors created by the TensorFlow native library and passed on the 
    Scala API (e.g., fetched from a TensorFlow session), we use a 
    combination of weak references and a disposing thread running in the 
    background. Please refer to 
    `tensorflow/src/main/scala/org/platanios/tensorflow/api/utilities/Disposer.scala`.
- Numpy-like indexing/slicing for tensors. For example:
  ```scala
  tensor(2 :: 5, ---, 1) // is equivalent to numpy's 'tensor[2:5, ..., 1]'
  ```
- Useful implicits that make using the library almost as simple as using 
  the Python API, which retaining type-safety (for the most part).

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

- Op Creation API:
  - [x] Graph op creation context
  - [x] Name scope op creation context
  - [x] Device op creation context
  - [x] Colocation op creation context
  - [x] Control dependencies op creation context
  - [x] Attribute op creation context
  - [x] Container op creation context
  - [x] Graph collections
  - [x] Variables support
  - [x] Gradients support
  - [ ] Kernel label map op creation context (may be unnecessary)
  - Ops:
    - [ ] Basic array ops (missing some)
    - [ ] Basic array ops gradients (missing most)
    - [ ] Math (missing some)
    - [ ] Math gradients (missing most)
    - [ ] NN
    - [ ] NN gradients
    - [ ] Control flow
    - [ ] Control flow gradients
    - [ ] Summaries
- Execution API:
  - [x] Default session
  - [ ] Session execution context (I'm not sure if that's good to have)
  - [ ] Session reset functionality
- Tensor API:
  - [ ] More flexible/efficient slicing for obtaining and assigning elements
  - [ ] More numpy-like operations for tensors
- General API Features:
  - [x] Slice creation
  - [x] Op output slicing
  - [ ] Variables slicing
  - [ ] Slice assignment
  - [x] Support for all data types
  - [x] Optimizers
  - [x] Savers
  - [ ] Estimators
  - [ ] tfprof / op statistics collection

## TODOs

- Improve support for creating string tensors.
- Revamp the Session API (e.g., using feed mapping implicits).
- Switch to using JUnit for all tests.
- Create a "Scope" class and companion object.
- Make casting more efficient with a conditional on the data type and an optional identity op.
- Add casting (considering type priorities) to the operator overloads.
- Variables API:
  - Clean up the implementation of variable scopes and stores and integrate it with "Scope".
  - Make 'PartitionedVariable' extend 'Variable'.
  - After that change, all 'getPartitionedVariable' methods can be integrated with the 'getVariable' methods, which will simplify the variables API.
  - Add tests.
- Switch to using "Seq" instead of "Array" wherever possible.
- Tensors:
  - Overloaded unary, binary, and comparison operators (data type aware)
  - Convenient string conversion methods
  - More efficient slicing (specialized contiguous slicing)
- Op creation:
  - Add tests for all of the op functions
  - Get graph from inputs
  - Assert same graph for inputs and control inputs
  - Support re-entering existing name scopes
  - Reset default graph
  - Register op statistics
- Fix Travis CI support (somehow load the native library)
