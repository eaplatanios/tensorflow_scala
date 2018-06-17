---
layout: page
title:  "Installation"
section: "installation"
position: 2
---

[sonatype]: https://oss.sonatype.org/index.html#nexus-search;quick~platanios

# Installation

TensorFlow for Scala is currently available for Scala 2.11.x and for 2.12.x. The main line of development is version 
2.12.6. Binary release artifacts are published to the [Sonatype OSS Repository Hosting service][sonatype] and synced to
Maven Central. Currently, given the beta status of this project, only snapshot releases are published.

## Using with SBT

To include the Sonatype repositories in your [SBT](http://www.scala-sbt.org) build and use TensorFlow for Scala, you 
should add the following to your `build.sbt` file:

```scala
libraryDependencies += "org.platanios" %% "tensorflow" % "0.2.1"
```

**NOTE:** This requires that you have installed the TensorFlow dynamic library in your system. If you haven't, please 
continue reading into the following section.

## Dependencies

### TensorFlow Dynamic Library

TensorFlow for Scala is an API for [TensorFlow](https://www.tensorflow.org). In order for it work, it requires that you 
have the main TensorFlow dynamic library installed. You have two options for dealing with this requirement:

  1. Use:
     
     ```scala
     libraryDependencies += "org.platanios" %% "tensorflow" % "0.2.1" classifier "linux-cpu-x86_64"
     ```
     
     instead of the line above, for your `build.sbt` file. *Make sure to replace `linux-cpu-x86_64` with the string 
     that corresponds to your platform.* Currently supported platforms are: `linux-cpu-x86_64`, `linux-gpu-x86_64`, and 
     `darwin-cpu-x86_64`. I hope to support Windows soon.
  2. Compile the TensorFlow dynamic library yourself and install it in your system. This is the recommended approach if 
     you care about performance, but it is also significantly more complicated and time consuming. It consists of the 
     following steps:
     
       1. Clone the TensorFlow repository:
       
          ```bash
          git clone https://github.com/tensorflow/tensorflow.git <repository_directory>
          cd <repository_directory>
          git checkout r1.9
          ```
          
       2. Compile the library using the following commands:
          
          ```bash
          ./configure
          bazel build --config=opt //tensorflow:libtensorflow.so
          ```
          
          For details regarding the configuration options (e.g., GPU support), please refer to the relevant main 
          TensorFlow [documentation page](https://www.tensorflow.org/install/install_sources).
       3. Copy the `bazel-bin/tensorflow/libtensorflow.so` (possibly having  a different extension, depending on the 
          platform you're using) file in a directory that is in `LD_LIBRARY_PATH`, or set `LD_LIBRARY_PATH` 
          appropriately.

**NOTE:** If you want to compile TensorFlow for Scala yourself and the `libtensorflow.so` file is not placed in one of 
the default system libraries directories, then set `-Djava.library.path=<directory>` (replacing `<directory>` with the 
directory containing the `libtensorflow.so` file) in the `.jvmopts` file at the root of the TensorFlow for Scala 
repository.

### Protocol Buffers Compiler

TensorFlow for Scala also requires `protoc`, the Protocol Buffers compiler, to be installed. You also have two options 
for dealing with this requirement:

  1. Install it using a package manager:
     - On Debian/Ubuntu, you can install it with [APT](https://help.ubuntu.com/community/AptGet), using the following 
       command:
       
       ```bash
       apt-get install protobuf-compiler
       ```
     - On Mac, you can install with [Homebrew](https://brew.sh), using the following command:
       
       ```bash
       brew install protobuf
       ```
  2. Download pre-built binaries from 
     [https://github.com/google/protobuf/releases/](https://github.com/google/protobuf/releases/) (choose the `protoc` 
     variant appropriate for your platform) and make sure that `protoc` is in the `PATH` (either by installing it in a 
     location in the `PATH`, or by adding its location to the `PATH`).

**NOTE:** You need to install `protoc` with version at least 3.
