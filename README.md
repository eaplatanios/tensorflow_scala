# TensorFlow API for Scala

[![Build Status](https://travis-ci.com/eaplatanios/tensorflow_scala.svg?token=VBPxqvcGXTuwbjkVyN68&branch=master)](https://travis-ci.com/eaplatanios/tensorflow_scala)

This is an attempt to replicate most of the TensorFlow Python API 
functionality in Scala. It is a work in progress and a project I started 
working on for my personal research purposes. The API is thus very 
likely to keep changing until I settle with something that makes me 
happy. Having said that, I try to stick as close as possible to the 
Python API usage to make transition easy for users and much of what I 
have already implemented is likely to stay as is. Most of the code is 
ported from the Python API with changes to make it more Scala-friendly 
and to make use of useful Scala features and strong-typing.

People who are new to TensorFlow should first go through the official 
Python API documentation at 
[https://www.tensorflow.org](https://www.tensorflow.org). Most of what 
you read there applies here too.

## Main Features

- Exposed API namespace similar to that of the Python API. For example:
  - `tf.constant(...)` creates a constant op.
  - `tf.reshape(...)` creates a reshape op.
  - `tf.Graph(...)` creates a graph.
  - `tf.Session(...)` creates a session.
  - etc.
- Straightforward API for graph creation. For example:
  ```scala
  import org.platanios.tensorflow.api._
  
  val inputs = tf.placeholder(FLOAT32, Shape(-1, 10))
  val outputs = tf.placeholder(FLOAT32, Shape(-1, 10))
  val predictions = tf.createWith(nameScope = "Linear") {
    val weights = tf.variable("weights", FLOAT32, Shape(10, 1), tf.zerosInitializer)
    val predictions = tf.matmul(inputs, weights)
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
  the Python API, while retaining type-safety (for the most part).
  
## How to Start

I would suggest you first follow the installation instructions, further 
down in this document, and then move on to look at the `examples` package 
and the unit tests of the main library. There is no user guide yet, as 
it's super early for this project, but knowing how to use the TensorFlow 
Python API and reading the code should be sufficient to get you started.

## High-Priority TODOs

It would be awesome if people could contribute to this library. Given 
its scope and its early state, before I settle on the API for some of 
the features, I would really appreciate contributions on the following:
- **Unit Tests:** Currently unit tests are missing for a big part of the 
  library and it would be extremely useful if we had those.
- **Op Implementations:** The process of implementing ops and their 
  gradients in the `org.platanios.tensorflow.api.ops` package is pretty 
  simple and self-explanatory by looking at `Basic.scala` and 
  `Math.scala`. It would be great if we could get better coverage of 
  the Python API ops. Porting them is simple, but tedious, and I plan 
  to do it mainly on an on-demand basis.
- **Examples:** Examples of code using the library would be great and 
  would also make issues come up early so they can be fixed.

## Installation

You first need to make sure you have the TensorFlow dynamic library 
installed. You can either download pre-compiled versions of it, or you 
can compile it yourself from the TensorFlow sources.

**NOTE:** *The pre-compiled versions that you can download are not 
currently supported. Please compile the native library yourself. The 
pre-compiled versions will start working again once their updated as 
the Scala API depends on some functionality that was added after the 
last release.*

### Downloading the TensorFlow Dynamic Library

You can download it from one of the following links:
  - **Linux:**
    - CPU-only: [https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.3.0.tar.gz](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.3.0.tar.gz)
    - GPU-enabled: [https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.3.0.tar.gz](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.3.0.tar.gz)
  - **Mac:**
    - CPU-only: [https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.3.0.tar.gz](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.3.0.tar.gz)
  - **Windows:** 
    - CPU-only: [https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.3.0.zip](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.3.0.zip)

### Compiling the TensorFlow Dynamic Library

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
     
     On ubuntu:
     ```bash
     sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib/
     ```
     
### Installing the Protocol Buffers Compiler

You also need protoc, the Protocol Buffers compiler.

On Debian/Ubuntu, you can install it with APT:

```bash
apt install protobuf-compiler
```
or

You can also download prebuild binaries from [https://github.com/google/protobuf/releases/](https://github.com/google/protobuf/releases/)
(choose the protoc variant appropriate for your platform).
```bash
# Make sure you grab the latest version
curl -OL https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-linux-x86_64.zip

# Unzip
unzip protoc-3.4.0-linux-x86_64.zip -d protoc3

# Move protoc to /usr/local/bin/
sudo mv protoc3/bin/* /usr/local/bin/

# Move protoc3/include to /usr/local/include/
sudo mv protoc3/include/* /usr/local/include/

# Optional: change owner
sudo chwon [user] /usr/local/bin/protoc
sudo chwon -R [user] /usr/local/include/google
```

### Compiling the TensorFlow Scala API

First make sure you have [CMake](https://cmake.org/install/) installed, 
and that the TensorFlow dynamic library is in the path. Then, you can  
compile the Scala API by running the following command from within the 
`tensorflow_scala` directory:

```bash
sbt compile
```

## Supported Features

- Learn API:
  - [ ] Design
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
    - [x] Basic array ops (missing quantization ops)
    - [x] Basic array ops gradients 
          (missing the concatenate op gradient for indexed slices -- need better Tensors API for that)
    - [x] Math
    - [x] NN (missing CNN ops)
    - [ ] RNN
    - [ ] Control flow
    - [ ] Control flow gradients
    - [x] IO
    - [ ] Summaries
- Execution API:
  - [x] Default session
  - [ ] Session execution context (I'm not sure if that's good to have)
  - [ ] Session reset functionality
- Tensor API:
  - [x] Support for the TensorFlow eager op execution API
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

## Some TODOs

- Find a way to automatically generate the duplicate documentation for symbolic and eager ops.
- Switch to using JUnit for all tests.
- Add convenience implicit conversions for shapes (e.g., from tuples or sequences of integers).
- Create a "Scope" class and companion object.
- Variables API:
  - Clean up the implementation of variable scopes and stores and integrate it with "Scope".
  - Make 'PartitionedVariable' extend 'Variable'.
  - After that change, all 'getPartitionedVariable' methods can be integrated with the 'getVariable' methods, which will simplify the variables API.
- Switch to using "Seq" instead of "Array" wherever possible.
- Tensors:
  - Overloaded unary, binary, and comparison operators (data type aware)
  - Convenient string conversion methods
- Op creation:
  - Add tests for all of the op functions
  - Get graph from inputs
  - Assert same graph for inputs and control inputs
  - Support re-entering existing name scopes
  - Reset default graph
  - Register op statistics
- Fix Travis CI support (somehow load the native library)
