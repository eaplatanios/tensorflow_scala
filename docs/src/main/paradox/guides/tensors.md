# Tensors

TensorFlow, as the name indicates, is a framework used to
define and run computations involving tensors. A tensor is
a generalization of vectors and matrices to potentially
higher dimensions. Internally, TensorFlow represents
tensors as `n`-dimensional arrays of some underlying data
type. A @scaladoc[Tensor](org.platanios.tensorflow.api.tensors.Tensor)
has a @scaladoc[DataType](org.platanios.tensorflow.api.core.types.DataType)
(e.g., `FLOAT32`, which corresponds to 32-bit floating
point numbers) and a
@scaladoc[Shape](org.platanios.tensorflow.api.core.Shape) (that
is, the number of dimensions it has and the size of each
dimension -- e.g., `Shape(10, 2)` which corresponds to a
matrix with 10 rows and 2 columns) associated with it. Each
element in the
@scaladoc[Tensor](org.platanios.tensorflow.api.tensors.Tensor) has
the same data type. For example, the following code creates
an integer tensor filled with zeros with shape `[2, 5]`
(i.e., a two-dimensional array holding integer values, where the
first dimension size is 2 and the second is 5):

@@snip [Tensors.scala](/docs/src/main/scala/Tensors.scala) { #zeros_tensor_example }

You can print the contents of a tensor as follows:

@@snip [Tensors.scala](/docs/src/main/scala/Tensors.scala) { #tensor_summarize_example }

## Tensor Creation

Tensors can be created using various constructors defined in
the @scaladoc[Tensor](org.platanios.tensorflow.api.tensors.Tensor)
companion object. For example:

@@snip [Tensors.scala](/docs/src/main/scala/Tensors.scala) { #tensor_creation_examples }

## Data Types

As already mentioned, tensors have a data type. Various
numeric data types are supported, as well as strings (i.e.,
tensors containing strings are supported). It is not
possible to have a
@scaladoc[Tensor](org.platanios.tensorflow.api.tensors.Tensor) with
more than one data type. It is possible, however, to
serialize arbitrary data structures as strings and store
those in tensors.

The list of all supported data types is:

```
STRING     // String
BOOLEAN    // Boolean
FLOAT16    // 16-bit half-precision floating-point
FLOAT32    // 32-bit single-precision floating-point
FLOAT64    // 64-bit double-precision floating-point
BFLOAT16   // 16-bit truncated floating-point
COMPLEX64  // 64-bit single-precision complex
COMPLEX128 // 128-bit double-precision complex
INT8       // 8-bit signed integer
INT16      // 16-bit signed integer
INT32      // 32-bit signed integer
INT64      // 64-bit signed integer
UINT8      // 8-bit unsigned integer
UINT16     // 16-bit unsigned integer
QINT8      // Quantized 8-bit signed integer
QINT16     // Quantized 16-bit signed integer
QINT32     // Quantized 32-bit signed integer
QUINT8     // Quantized 8-bit unsigned integer
QUINT16    // Quantized 16-bit unsigned integer
RESOURCE   // Handle to a mutable resource
VARIANT    // Variant
```

TensorFlow Scala also provides value classes for the types
that are not natively supported by Scala (e.g.,
@scaladoc[UByte](org.platanios.tensorflow.api.core.types.UByte)
corresponds to `UINT8`).

It is also possible to cast tensors from one data type to
another using the `toXXX` operator, or the `castTo[XXX]`
operator:

@@snip [Tensors.scala](/docs/src/main/scala/Tensors.scala) { #tensor_cast_examples }

A tensor's data type can be inspected using:

@@snip [Tensors.scala](/docs/src/main/scala/Tensors.scala) { #tensor_datatype_example }

@@@ warning { title='Performing Operations on Tensors' }

In general, all tensor-supported operations can be accessed
as direct methods/operators of the
@scaladoc[Tensor](org.platanios.tensorflow.api.tensors.Tensor)
object, or as static methods defined in the
@scaladoc[tfi](org.platanios.tensorflow.api.tfi) package,
which stands for *TensorFlow Imperative*
(given the imperative nature of this API).

@@@

## Shape

The shape of a tensor is the number of elements it contains
in each dimension. The TensorFlow documentation uses two
notational conventions to describe tensor dimensionality:
*rank*, and *shape*. The following table shows how these
relate to one another:

| Rank | Shape              | Example                                 |
|:-----|:-------------------|:----------------------------------------|
| 0    | []                 | A 0-D tensor. A scalar.                 |
| 1    | [D0]               | A 1-D tensor with shape [5].            |
| 2    | [D0, D1]           | A 2-D tensor with shape [3, 4].         |
| 3    | [D0, D1, D2]       | A 3-D tensor with shape [1, 4, 3].      |
| n    | [D0, D1, ... Dn-1] | A tensor with shape [D0, D1, ... Dn-1]. |

Shapes can be automatically converted to integer tensors,
if necessary.

@@@ note

Shapes are automatically converted to `Tensor[Int]` and not
`Tensor[Long]` in order to improve performance when working
with GPUs. The reason is that TensorFlow treats integer
tensors in a special manner, if they are placed on GPUs,
assuming that they represent shapes.

@@@

For example:

@@snip [Tensors.scala](/docs/src/main/scala/Tensors.scala) { #tensor_shape_examples }

The shape of a tensor can be inspected using:

@@snip [Tensors.scala](/docs/src/main/scala/Tensors.scala) { #tensor_inspect_shape_example }

### Rank

The rank of of a tensor is its number of dimensions.
Synonyms for rank include order or degree or `n`-dimension.
Note that rank in TensorFlow is not the same as matrix rank
in mathematics. As the following table shows, each rank in
TensorFlow corresponds to a different mathematical entity:

| Rank | Math Entity                      |
|:-----|:---------------------------------|
| 0    | Scalar (magnitude only)          |
| 1    | Vector (magnitude and direction) |
| 2    | Matrix (table of numbers)        |
| 3    | 3-Tensor (cube of numbers)       |
| n    | n-Tensor (you get the idea)      |

The rank of a tensor can be inspected using:

@@snip [Tensors.scala](/docs/src/main/scala/Tensors.scala) { #tensor_inspect_rank_example }

## Indexing / Slicing

Similar to NumPy, tensors can be indexed/sliced in various
ways. An indexer can be one of:

  - `Ellipsis`: Full slice over multiple dimensions of a
    tensor. Ellipses are used to represent zero or more
    dimensions of a full-dimension indexer sequence.
  - `NewAxis`: Addition of a new dimension.
  - `Slice`: Slice over a single dimension of a tensor.

Examples of constructing and using indexers are provided in
the @scaladoc[Ellipsis](org.platanios.tensorflow.api.core.Ellipsis)
and the @scaladoc[Slice](org.platanios.tensorflow.api.core.Slice)
documentation. Here we provide examples of indexing over
tensors using indexers:

@@snip [Tensors.scala](/docs/src/main/scala/Tensors.scala) { #tensor_indexer_examples }

where `---` corresponds to an ellipsis.

Note that each indexing sequence is only allowed to contain
at most one ellipsis. Furthermore, if an ellipsis is not
provided, then one is implicitly appended at the end of the
indexing sequence. For example, `foo(2 :: 4)` is equivalent
to `foo(2 :: 4, ---)`.
