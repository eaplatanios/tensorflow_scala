---
layout: docs
title: "Getting Started"
section: "getting_started"
position: 3
---

```tut:invisible
import org.platanios.tensorflow.api._
```

[tensor]: /tensorflow_scala/api/org/platanios/tensorflow/api/tensors/Tensor.html
[output]: /tensorflow_scala/api/org/platanios/tensorflow/api/ops/Output.html
[data_type]: /tensorflow_scala/api/types/DataType.html
[shape]: /tensorflow_scala/api/core/Shape.html
[variable]: /tensorflow_scala/api/ops/variables/Variable.html

[tf_python]: https://www.tensorflow.org/get_started/get_started

# Introduction

Similar to the [TensorFlow Python API][tf_python], by Google, TensorFlow for Scala provides multiple APIs. The lowest
level API -- TensorFlow Core -- provides you with complete programming control. TensorFlow Core is suitable for machine
learning researchers and others who require fine levels of control over their models. The higher level APIs are built on
top of TensorFlow Core. These higher level APIs are typically easier to learn and use than TensorFlow Core. In addition,
the higher level APIs make repetitive tasks easier and more consistent between different users. A high-level API like
the Learn API (which is currently under development) helps you manage data sets, models, training, and inference.

The main APIs of TensorFlow for Scala introduced in this guide are:

  - **[Tensor API:](#tensors-1)** Provides a simple way for manipulating tensors and performing computations involving 
    tensors. This is similar in functionality to the [NumPy](http://www.numpy.org) library used by Python programmers.
  - **[Learn API:](#neural-networks-2)** High-level interface for creating, training, and using neural networks. This is 
    similar in functionality to the [Keras](https://keras.io) library used by Python programmers, with the main 
    difference being that it is strongly-typed and offers a much richer functional interface for building neural 
    networks. Furthermore, it supports distributed training in a way that is very similar to the 
    [TensorFlow Estimators API](https://www.tensorflow.org/programmers_guide/estimators).
  - **[Core API:](#core-3)** Low-level graph construction interface, similar to that offered by the TensorFlow 
    Python API, with the main difference being that this interface is strongly-typed wherever possible.

The fact that this library is strongly-typed is mentioned a couple times in the above paragraph and that's because it is 
a very important feature. It means that many problems with the code you write will show themselves at compile time, 
which means that your chances of running into the experience of waiting for a neural network to train for a week only to 
find out that your evaluation code crashed and you lost everything, decrease significantly.

This guide starts with an introduction of the **[Tensor API](#tensors-1)** and goes from high-level to low-level 
concepts as you progress. Concepts such as the TensorFlow graph and sessions only appear in the **[Core API](#core-3)** 
section.

**NOTE:** This guide borrows a lot pf material from the official [Python API documentation][tf_python] of TensorFlow and 
adapts it for the purposes of TensorFlow for Scala. It also introduces a lot of new constructs specific to this library.

# Tensors

TensorFlow, as the name indicates, is a framework to define and run computations involving tensors. A tensor is a 
generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as 
`n`-dimensional arrays of some underlying data type. A [`Tensor`][tensor] has a [`DataType`][data_type] (e.g., `FLOAT32` 
which corresponds to 32-bit floating point numbers) and a [`Shape`][shape] (that is, the number of dimensions it has and 
the size of each dimension -- e.g., `Shape(10, 2)` which corresponds to a matrix with 10 rows and 2 columns) associated 
with it. Each element in the [`Tensor`][tensor] has the same data type.

For example, the following code creates an integer tensor filled with 
zeros with shape `[2, 5]` (i.e., a two-dimensional array holding integer values, where the first dimension size is 2 and 
the second is 5):
```tut
val tensor = Tensor.zeros(INT32, Shape(2, 5))
```
You can print the contents of a tensor as follows:
```tut
tensor.summarize()
```

## Data Type

As already mentioned, tensors have a data type. Various numeric data types are supported, as well as strings (i.e., 
tensors containing strings are supported). It is not possible to have a [`Tensor`][tensor] with more than one data type. 
It is possible, however, to serialize arbitrary data structures as strings and store those in [`Tensor`][tensor]s.

The list of all supported data types is:

  - **STRING:** String.
  - **BOOLEAN:** Boolean.
  - **FLOAT16:** 16-bit half-precision floating-point.
  - **FLOAT32:** 32-bit single-precision floating-point.
  - **FLOAT64:** 64-bit double-precision floating-point.
  - **BFLOAT16:** 16-bit truncated floating-point.
  - **COMPLEX64:** 64-bit single-precision complex.
  - **COMPLEX128:** 128-bit double-precision complex.
  - **INT8:** 8-bit signed integer.
  - **INT16:** 16-bit signed integer.
  - **INT32:** 32-bit signed integer.
  - **INT64:** 64-bit signed integer.
  - **UINT8:** 8-bit unsigned integer.
  - **UINT16:** 16-bit unsigned integer.
  - **QINT8:** Quantized 8-bit signed integer.
  - **QINT16:** Quantized 16-bit signed integer.
  - **QINT32:** Quantized 32-bit signed integer.
  - **QUINT8:** Quantized 8-bit unsigned integer.
  - **QUINT16:** Quantized 16-bit unsigned integer.
  - **RESOURCE:** Handle to a mutable resource.
  - **VARIANT:** Variant.

It is also possible to cast [`Tensor`][tensor]s from one data type to another using the `cast` operator:
```tut:silent
val floatTensor = Tensor(FLOAT32, 1, 2, 3) // Floating point vector containing the elements: 1.0f, 2.0f, and 3.0f
floatTensor.cast(INT32)                    // Integer vector containing the elements: 1, 2, and 3
tfi.cast(floatTensor, INT32)               // Integer vector containing the elements: 1, 2, and 3
```

**NOTE:** In general, all tensor-supported operations can be accessed as direct methods/operators of the 
[`Tensor`][tensor] object, or as static methods defined in the `tfi` package, which stands for *TensorFlow Imperative* 
(given the imperative nature of that API).

A [`Tensor`][tensor]'s data type can be inspected using:
```tut:silent
floatTensor.dataType // Returns FLOAT32
```

When creating a [`Tensor`][tensor] from a Scala objects you may optionally specify the data type. If you don't, 
TensorFlow chooses a data type that can represent your data. It converts Scala integers to `INT32` and Scala floating 
point numbers to either `FLOAT32` or `FLOAT64` depending on their precision.

```tut:silent
Tensor(1, 2, 3)      // INT32 tensor
Tensor(1, 2L, 3)     // INT64 tensor
Tensor(2.4f, -0.1f)  // FLOAT32 tensor
Tensor(0.6f, 1.0)    // FLOAT64 tensor
```

## Rank

The rank of a [`Tensor`][tensor] is its number of dimensions. Synonyms for rank include order or degree or 
`n`-dimension. Note that rank in TensorFlow is not the same as matrix rank in mathematics. As the following table shows, 
each rank in TensorFlow corresponds to a different mathematical entity:

| Rank | Math Entity                      |
|:-----|:---------------------------------|
| 0    | Scalar (magnitude only)          |
| 1    | Vector (magnitude and direction) |
| 2    | Matrix (table of numbers)        |
| 3    | 3-Tensor (cube of numbers)       |
| n    | n-Tensor (you get the idea)      |

For example:
```tut:silent
val t0 = Tensor.ones(INT32, Shape())     // Creates a scalar equal to the value 1
val t1 = Tensor.ones(INT32, Shape(10))   // Creates a vector with 10 elements, all of which are equal to 1
val t2 = Tensor.ones(INT32, Shape(5, 2)) // Creates a matrix with 5 rows with 2 columns

// You can also create tensors in the following way:
val t3 = Tensor(2.0, 5.6)                                 // Creates a vector that contains the numbers 2.0 and 5.6
val t4 = Tensor(Tensor(1.2f, -8.4f), Tensor(-2.3f, 0.4f)) // Creates a matrix with 2 rows and 2 columns
```

A rank of a tensor can be obtained in one of two ways:
```tut:silent
t4.rank      // Returns the value 2
tfi.rank(t4) // Also returns the value 2
```

## Shape


## Indexing / Slicing

Similar to NumPy, tensors can be indexed/sliced in various ways:

An indexer can be one of:
  - `Ellipsis`: Corresponds to a full slice over multiple dimensions of a tensor. Ellipses are used to represent
    zero or more dimensions of a full-dimension indexer sequence.
  - `NewAxis`: Corresponds to the addition of a new dimension.
  - `Slice`: Corresponds to a slice over a single dimension of a tensor.

Examples of constructing and using indexers are provided in the `Ellipsis` and the `Slice` class documentation.
Here we provide examples of indexing over tensors using indexers:
```tut:silent
val t = Tensor.zeros(FLOAT32, Shape(4, 2, 3, 8))
t(::, ::, 1, ::)            // Tensor with shape [4, 2, 1, 8]
t(1 :: -2, ---, 2)          // Tensor with shape [1, 2, 3, 1]
t(---)                      // Tensor with shape [4, 2, 3, 8]
t(1 :: -2, ---, NewAxis, 2) // Tensor with shape [1, 2, 3, 1, 1]
t(1 ::, ---, NewAxis, 2)    // Tensor with shape [3, 2, 3, 1, 1]
```
where `---` corresponds to an ellipsis.

Note that each indexing sequence is only allowed to contain at most one Ellipsis. Furthermore, if an ellipsis is not 
provided, then one is implicitly appended at the end of indexing sequence. For example, `foo(2 :: 4)` is equivalent to 
`foo(2 :: 4, ---)`.

# Neural Networks


# Data


# Core

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
