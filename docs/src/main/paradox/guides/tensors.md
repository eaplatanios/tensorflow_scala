# Tensors

TensorFlow, as the name indicates, is a framework to define and run computations involving tensors. A tensor is a
generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as
`n`-dimensional arrays of some underlying data type. A @scaladoc[Tensor](org.platanios.tensorflow.api.Tensor) has a @scaladoc[DataType](org.platanios.tensorflow.api.DataType) (e.g., `FLOAT32`
which corresponds to 32-bit floating point numbers) and a @scaladoc[Shape](org.platanios.tensorflow.api.Shape) (that is, the number of dimensions it has and
the size of each dimension -- e.g., `Shape(10, 2)` which corresponds to a matrix with 10 rows and 2 columns) associated
with it. Each element in the @scaladoc[Tensor](org.platanios.tensorflow.api.Tensor) has the same data type. For example, the following code creates an
integer tensor filled with zeros with shape `[2, 5]` (i.e., a two-dimensional array holding integer values, where the
first dimension size is 2 and the second is 5):

```scala
val tensor = Tensor.zeros[Int](Shape(2, 5))
```
You can print the contents of a tensor as follows:
```scala
tensor.summarize(flattened = true)
```

## Tensor Creation

Tensors can be created using various constructors defined in the `Tensor`
companion object. For example:

```scala
val a = Tensor(1, 2)      // Creates a Tensor[Int] with shape [2]
val b = Tensor(1L, 2)     // Creates a Tensor[Long] with shape [2]
val c = Tensor(3.0f)      // Creates a Tensor[Float] with shape [1]
val d = Tensor(-4.0)      // Creates a Tensor[Double] with shape [1]
val e = Tensor.empty[Int] // Creates an empty Tensor[Int] with shape [0]
val z = Tensor.zeros[Float](Shape(5, 2))   // Creates a zeros Tensor[Float] with shape [5, 2]
val r = Tensor.randn(Double, Shape(10, 3)) // Creates a Tensor[Double] with shape [10, 3] and
                                           // elements drawn from the standard Normal distribution.
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

It is also possible to cast [`Tensor`][tensor]s from one data type to another using the `toXXX` operator, or the
`castTo[XXX]` operator:
```scala
val floatTensor = Tensor[Float](1, 2, 3) // Floating point vector containing the elements: 1.0f, 2.0f, and 3.0f.
floatTensor.toInt                        // Integer vector containing the elements: 1, 2, and 3.
floatTensor.castTo[Int]                  // Integer vector containing the elements: 1, 2, and 3.
```

**NOTE:** In general, all tensor-supported operations can be accessed as direct methods/operators of the
[`Tensor`][tensor] object, or as static methods defined in the `tfi` package, which stands for *TensorFlow Imperative*
(given the imperative nature of that API).

A [`Tensor`][tensor]'s data type can be inspected using:
```scala
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

```scala
val t = Tensor.zeros[Float](Shape(4, 2, 3, 8))
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
