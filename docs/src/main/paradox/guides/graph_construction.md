# Graph Construction

The low level API can be used to define computations that will be executed at a later point, and potentially execute
them. It can also be used to create custom layers for the [Learn API](#neural-networks-2). The main type of object
underlying the low level API is the [`Output`][output], which represents the value of a [`Tensor`][tensor] that has not
yet been computed. Its name comes from the fact that it represents the *output* of some computation. An
[`Output`][output] object thus represents a partially defined computation that will eventually produce a value. Core
TensorFlow programs work by first building a graph of [`Output`][output] objects, detailing how each output is computed
based on the other available outputs, and then by running parts of this graph to achieve the desired results.

Similar to a [`Tensor`][tensor], each element in an [`Output`][output] has the same data type, and the data type is
always known. However, the shape of an [`Output`][output] might be only partially known. Most operations produce tensors
of fully-known shapes if the shapes of their inputs are also fully known, but in some cases it's only possible to find
the shape of a tensor at graph execution time.

It is important to understand the main concepts underlying the core API:

  - **Tensor:**
  - **Output:**
    - **Sparse Output:**
    - **Placeholder:**
    - **Variable:**
  - **Graph:**
  - **Session:**

With the exception of [`Variable`][variable]s, the value of outputs is immutable, which means that in the context of a
single execution, outputs only have a single value. However, evaluating the same output twice can result in different
values. For example, that tensor may be the result of reading data from disk, or generating a random number.

## Graph


## Working with Outputs


### Evaluating Outputs


### Printing Outputs


### Logging

Logging in the native TensorFlow library can be controlled by setting the `TF_CPP_MIN_LOG_LEVEL` environment variable:

  - `0`: Debug level (default).
  - `1`: Warning level.
  - `2`: Error level.
  - `3`: Fatal level.
