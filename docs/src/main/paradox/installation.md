# Installation

TensorFlow for Scala is currently available for Scala 
2.12 and for 2.13. The main line of development is
version 2.12.11. Binary release artifacts are published to
the [Sonatype OSS Repository Hosting service](https://oss.sonatype.org/index.html#nexus-search;quick~platanios) 
and synced to Maven Central. Currently, given the beta 
status of this project, only snapshot releases are 
published.

## Library Dependencies

To include the Sonatype repositories in your 
[SBT](http://www.scala-sbt.org) build and use TensorFlow 
for Scala, you should add the following dependency:

@@ dependency[sbt,Maven,Gradle] {
  group="org.platanios"
  artifact="tensorflow"
  version="0.4.1"
}

@@@ note

This requires that you have installed the TensorFlow
dynamic library in your system. If you haven't, please
continue reading into the following section.

@@@

## Dependencies

### TensorFlow Dynamic Library

TensorFlow for Scala is an API for 
[TensorFlow](https://www.tensorflow.org). In order for it 
work, it requires that you have the main TensorFlow dynamic 
library installed. You have two options for dealing with 
this requirement:

#### Using Precompiled Binaries

Add the following dependency, instead of the previous one:

@@ dependency[sbt,Maven,Gradle] {
  group="org.platanios"
  artifact="tensorflow"
  version="0.4.1"
  classifier="linux-cpu-x86_64"
}

@@@ warning { title='Operating System' }

Make sure to replace `linux-cpu-x86_64` with the string
that corresponds to your platform.* Currently supported
platforms are: `linux-cpu-x86_64`, `linux-gpu-x86_64`, and
`darwin-cpu-x86_64`.

@@@

#### Compiling TensorFlow from Scratch

Compile the TensorFlow dynamic library yourself and install
it in your system. This is the recommended approach if you
care about performance, but it is also significantly more
complicated and time consuming.

First, clone the TensorFlow repository:

@@snip [installation.sh](/docs/src/main/scala/installation.sh) { #clone_repository }

Then, compile TensorFlow using the following commands:

@@snip [installation.sh](/docs/src/main/scala/installation.sh) { #compile_tf }

For details regarding the configuration options (e.g., GPU
support), please refer to the relevant main TensorFlow
[documentation page](https://www.tensorflow.org/install/install_sources).

Finally, copy the `bazel-bin/tensorflow/libtensorflow.so`
file (possibly having a different extension, depending on
the platform you're using) file in a directory that is in
`LD_LIBRARY_PATH`, or set `LD_LIBRARY_PATH` appropriately.

@@@ warning { title='Compiling TensorFlow Scala' }

If you want to compile TensorFlow for Scala
yourself and the `libtensorflow.so` file is not placed in
one of the default system libraries directories, then set
`-Djava.library.path=<directory>` (replacing `<directory>`
with the directory containing the `libtensorflow.so` file)
in the `.jvmopts` file at the root of the TensorFlow for
Scala repository.

@@@

### Protocol Buffers Compiler

TensorFlow for Scala also requires `protoc`, the Protocol
Buffers compiler (at least version 3), to be installed. You
also have two options for dealing with this requirement.

#### Using Precompiled Binaries

Download pre-built binaries from
[https://github.com/google/protobuf/releases/](https://github.com/google/protobuf/releases/)
(choose the `protoc` variant appropriate for your platform)
and make sure that `protoc` is in the `PATH` (either by
installing it in a location in the `PATH`, or by adding its
location to the `PATH`).

#### Installing Using a Package Manager

Install it using a package manager:

  - On Debian/Ubuntu, you can install it with
    [APT](https://help.ubuntu.com/community/AptGet), using
    the following command:

    @@snip [installation.sh](/docs/src/main/scala/installation.sh) { #apt_get_install_protobuf }

  - On Mac, you can install with
    [Homebrew](https://brew.sh), using the following
    command:

    @@snip [installation.sh](/docs/src/main/scala/installation.sh) { #brew_install_protobuf }
