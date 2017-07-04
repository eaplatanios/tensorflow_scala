/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
trait Math {
  /** Creates an op that selects elements from `x` or `y`, depending on `condition`.
    *
    * The `x`, and `y` tensors must have the same shape. The output tensor will also have the same shape.
    *
    * The `condition` tensor must be a scalar if `x` and `y` are scalars. If `x` and `y` are vectors or higher rank,
    * then `condition` must be either a scalar, or a vector with size matching the first dimension of `x`, or it must
    * have the same shape as `x`.
    *
    * The `condition` tensor acts as a mask that chooses, based on the value at each element, whether the corresponding
    * element / row in the output should be taken from `x` (if true) or `y` (if false).
    *
    * If `condition` is a vector and `x` and `y` are higher rank matrices, then it chooses which row (outer dimension)
    * to copy from `x` and `y`. If `condition` has the same shape as `x` and `y`, then it chooses which element to copy
    * from `x` and `y`.
    *
    * For example:
    * {{{
    *   // 'condition' tensor is [[true,  false], [false, true]]
    *   // 'x' is [[1, 2], [3, 4]]
    *   // 'y' is [[5, 6], [7, 8]]
    *   select(condition, x, y) == [[1, 6], [7, 4]]
    *
    *   // 'condition' tensor is [true, false]
    *   // 'x' is [[1, 2], [3, 4]]
    *   // 'y' is [[5, 6], [7, 8]]
    *   select(condition, x, y) == [[1, 2], [7, 8]]
    * }}}
    *
    * @param  condition Boolean condition tensor.
    * @param  x         Tensor which may have the same shape as `condition`. If `condition` has rank `1`, then `t` may
    *                   have a higher rank, but its first dimension must match the size of `condition`.
    * @param  y         Tensor with the same data type and shape as `t`.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def select(condition: Output[DataType], x: Output[DataType], y: Output[DataType], name: String = "Select"): Output[DataType] = {
    Op.Builder(opType = "Select", name = name)
        .addInput(condition)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that constructs a diagonal tensor using the provided diagonal values.
    *
    * Given a `diagonal`, this op returns a tensor with that `diagonal` and everything else padded with zeros. The
    * diagonal is computed as follows:
    *
    * Assume that `diagonal` has shape `[D1,..., DK]`. Then the output tensor, `output`, is a rank-`2K` tensor with
    * shape `[D1, ..., DK, D1, ..., DK]`, where `output(i1, ..., iK, i1, ..., iK) = diagonal(i1, ..., iK)` and `0`
    * everywhere else.
    *
    * For example:
    * {{{
    *   // Tensor 'diagonal' is [1, 2, 3, 4]
    *   diag(diagonal) == [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
    * }}}
    *
    * This op is the inverse of [[diagPart]].
    *
    * @param  diagonal Diagonal values, represented as a rank-`K` tensor, where `K` can be at most `3`.
    * @param  name     Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `diagonal` has rank higher than `3`.
    */
  @throws[IllegalArgumentException]
  def diag(diagonal: Output[DataType], name: String = "Diag"): Output[DataType] = {
    if (diagonal.rank > 3)
      throw new IllegalArgumentException(s"The provided tensor (rank = ${diagonal.rank}) can have rank at most 3.")
    Op.Builder(opType = "Diag", name = name)
        .addInput(diagonal)
        .build().outputs(0)
  }

  /** Creates an op that returns the diagonal part of a tensor.
    *
    * This op returns a tensor with the `diagonal` part of the `input`. The `diagonal` part is computed as follows:
    *
    * Assume `input` has shape `[D1, ..., DK, D1, ..., DK]`. Then the output is a rank-`K` tensor with shape
    * `[D1,..., DK]`, where `diagonal(i1, ..., iK) = output(i1, ..., iK, i1, ..., iK)`.
    *
    * For example:
    * {{{
    *   // Tensor 'input' is [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
    *   diagPart(input) == [1, 2, 3, 4]
    * }}}
    *
    * This op is the inverse of [[diag]].
    *
    * @param  input Rank-`K` input tensor, where `K` is either `2`, `4`, or `6`.
    * @param  name  Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `input` has rank other than `2`, `4`, or `6`.
    */
  @throws[IllegalArgumentException]
  def diagPart[T <: DataType](input: Output[T], name: String = "DiagPart"): Output[T] = {
    if (input.rank != 2 && input.rank != 4 && input.rank != 6)
      throw new IllegalArgumentException(s"The provided tensor (rank = ${input.rank}) can only be 2, 4, or 6.")
    Op.Builder(opType = "DiagPart", name = name)
        .addInput(input)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that returns a batched diagonal tensor with the provided batched diagonal values.
    *
    * Given a `diagonal`, the op returns a tensor with that `diagonal` and everything else padded with zeros. Assuming
    * that `diagonal` has `k` dimensions `[I, J, K, ..., N]`, the output is a tensor of rank `k + 1` with dimensions
    * `[I, J, K, ..., N, N]`, where: `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.
    *
    * For example:
    * {{{
    *   // 'diagonal' is a tensor containing the values: [[1, 2, 3, 4], [5, 6, 7, 8]] (shape = [2, 4])
    *   tf.matrixDiag(diagonal) ==> [[[1, 0, 0, 0]
    *                                 [0, 2, 0, 0]
    *                                 [0, 0, 3, 0]
    *                                 [0, 0, 0, 4]],
    *                                [[5, 0, 0, 0]
    *                                 [0, 6, 0, 0]
    *                                 [0, 0, 7, 0]
    *                                 [0, 0, 0, 8]]]  // with shape [2, 4, 4]
    * }}}
    *
    * @param  diagonal Rank-`K` input tensor, where `K >= 1`.
    * @param  name     Name for the created op.
    * @return Created op output with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its last
    *         dimension duplicated.
    * @throws IllegalArgumentException If `input` has rank higher than `K < 1`.
    */
  @throws[IllegalArgumentException]
  def matrixDiag(diagonal: Output[DataType], name: String = "MatrixDiag"): Output[DataType] = {
    if (diagonal.rank > -1 && diagonal.rank < 1)
      throw new IllegalArgumentException(s"The provided tensor (rank = ${diagonal.rank}) must have rank at least 1.")
    Op.Builder(opType = "MatrixDiag", name = name)
        .addInput(diagonal)
        .build().outputs(0)
  }

  /** Creates an op that returns a batched matrix tensor with new batched diagonal values.
    *
    * Given `input` and `diagonal`, the op returns a tensor with the same shape and values as `input`, except for the
    * main diagonal of its innermost matrices. These diagonals will be overwritten by the values in `diagonal`. Assuming
    * that `input` has `k + 1` dimensions, `[I, J, K, ..., M, N]`, and `diagonal` has `k` dimensions,
    * `[I, J, K, ..., min(M, N)]`, then the output is a tensor of rank `k + 1` with dimensions `[I, J, K, ..., M, N]`,
    * where:
    *   - `output[i, j, k, ..., m, n] == diagonal[i, j, k, ..., n]`, for `m == n`, and
    *   - `output[i, j, k, ..., m, n] == input[i, j, k, ..., m, n]`, for `m != n`.
    *
    * @param  input    Rank-`K+1` tensor, where `K >= 1`.
    * @param  diagonal Rank-`K` tensor, where `K >= 1`.
    * @param  name     Name for the created op.
    * @return Created op output with rank equal to `K + 1` and shape equal to the shape of `input`.
    * @throws IllegalArgumentException If `input` has rank `K < 2`, or `diagonal` has rank `K < 1`.
    */
  @throws[IllegalArgumentException]
  def matrixSetDiag[T <: DataType](input: Output[T], diagonal: Output[DataType], name: String = "MatrixSetDiag"): Output[T] = {
    if (input.rank > -1 && input.rank < 2)
      throw new IllegalArgumentException(s"The provided input tensor (rank = ${input.rank}) must have rank at least 2.")
    if (diagonal.rank > -1 && diagonal.rank < 1)
      throw new IllegalArgumentException(
        s"The provided diagonal tensor (rank = ${diagonal.rank}) must have rank at least 1.")
    Op.Builder(opType = "MatrixSetDiag", name = name)
        .addInput(input)
        .addInput(diagonal)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that returns the batched diagonal part of a batched tensor.
    *
    * The op returns a tensor with the `diagonal` part of the batched `input`. Assuming that `input` has `k` dimensions,
    * `[I, J, K, ..., M, N]`, then the output is a tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]`,
    * where `diagonal[i, j, k, ..., n] == input[i, j, k, ..., n, n]`.
    *
    * Note that `input` must have rank of at least `2`.
    *
    * For example:
    * {{{
    *   // 'input' is a tensor containing the values:
    *   //   [[[1, 0, 0, 0]
    *   //     [0, 2, 0, 0]
    *   //     [0, 0, 3, 0]
    *   //     [0, 0, 0, 4]],
    *   //    [[5, 0, 0, 0]
    *   //     [0, 6, 0, 0]
    *   //     [0, 0, 7, 0]
    *   //     [0, 0, 0, 8]]]  with shape [2, 4, 4]
    *   tf.matrixDiagPart(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]  // with shape [2, 4]
    * }}}
    *
    * @param  input Rank-`K` tensor, where `K >= 2`.
    * @param  name  Name for the created op.
    * @return Created op output containing the diagonal(s) and having shape equal to
    *         `input.shape[:-2] + [min(input.shape[-2:])]`.
    * @throws IllegalArgumentException If `input` has rank `K < 2`.
    */
  @throws[IllegalArgumentException]
  def matrixDiagPart[T <: DataType](input: Output[T], name: String = "MatrixDiagPart"): Output[T] = {
    if (input.rank > -1 && input.rank < 2)
      throw new IllegalArgumentException(s"The provided input tensor (rank = ${input.rank}) must have rank at least 2.")
    Op.Builder(opType = "MatrixDiagPart", name = name)
        .addInput(input)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that copies a tensor, while setting everything outside a central band in each innermost matrix of
    * the tensor, to zero.
    *
    * Assuming that `input` has `k` dimensions, `[I, J, K, ..., M, N]`, the output is a tensor with the same shape,
    * where `band[i, j, k, ..., m, n] == indicatorBand(m, n) * input[i, j, k, ..., m, n]`. The indicator function is
    * defined as:
    * {{{
    *   indicatorBand(m, n) = (numSubDiagonals < 0 || m - n <= numSubDiagonals) &&
    *                         (numSuperDiagonals < 0 || n - m <= numSuperDiagonals)
    * }}}
    *
    * For example:
    * {{{
    *   // 'input' is a tensor containing the values:
    *   //   [[ 0,  1,  2, 3]
    *   //    [-1,  0,  1, 2]
    *   //    [-2, -1,  0, 1]
    *   //    [-3, -2, -1, 0]]
    *   tf.matrixBandPart(input, 1, -1) ==> [[ 0,  1,  2, 3]
    *                                        [-1,  0,  1, 2]
    *                                        [ 0, -1,  0, 1]
    *                                        [ 0,  0, -1, 0]]
    *   tf.matrixBandPart(input, 2, 1) ==>  [[ 0,  1,  0, 0]
    *                                        [-1,  0,  1, 0]
    *                                        [-2, -1,  0, 1]
    *                                        [ 0, -2, -1, 0]]
    * }}}
    *
    * Useful special cases:
    * {{{
    *   tf.matrixBandPart(input, 0, -1) ==> Upper triangular part
    *   tf.matrixBandPart(input, -1, 0) ==> Lower triangular part
    *   tf.matrixBandPart(input, 0, 0)  ==> Diagonal
    * }}}
    *
    * @param  input             Input tensor.
    * @param  numSubDiagonals   Scalar `INT64` tensor that contains the number of sub-diagonals to keep. If negative,
    *                           the entire lower triangle is kept.
    * @param  numSuperDiagonals Scalar `INT64` tensor that contains the number of super-diagonals to keep. If negative,
    *                           the entire upper triangle is kept.
    * @param  name              Name for the created op.
    * @return Created op output which contains the expected banded tensor and has rank `K` and same shape as `input`.
    * @throws IllegalArgumentException If `numSubDiagonals` or `numSuperDiagonals` are not scalar, or if their data type
    *                                  is not `INT64`.
    */
  @throws[IllegalArgumentException]
  def matrixBandPart[T <: DataType](
      input: Output[T], numSubDiagonals: Output[DataType], numSuperDiagonals: Output[DataType], name: String = "MatrixBandPart"): Output[T] = {
    if (numSubDiagonals.rank != 0 || numSubDiagonals.dataType != INT64)
      throw new IllegalArgumentException(
        s"'numSubDiagonals' (rank = ${numSubDiagonals.rank}, dataType = ${numSubDiagonals.dataType}) " +
            s"must be a scalar INT64 tensor.")
    if (numSuperDiagonals.rank != 0 || numSuperDiagonals.dataType != INT64)
      throw new IllegalArgumentException(
        s"'numSuperDiagonals' (rank = ${numSuperDiagonals.rank}, dataType = ${numSuperDiagonals.dataType}) " +
            s"must be a scalar INT64 tensor.")
    Op.Builder(opType = "MatrixBandPart", name = name)
        .addInput(input)
        .addInput(numSubDiagonals)
        .addInput(numSuperDiagonals)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that constructs a sequence of numbers.
    *
    * The op creates a sequence of numbers that begins at `start` and extends by increments of `delta` up to but not
    * including `limit`. The data type of the resulting tensor is inferred from the inputs unless it is provided
    * explicitly.
    *
    * For example:
    * {{{
    *   // 'start' is 3
    *   // 'limit' is 18
    *   // 'delta' is 3
    *   range(start, limit, delta) == [3, 6, 9, 12, 15]
    *
    *   // 'start' is 3
    *   // 'limit' is 1
    *   // 'delta' is -0.5
    *   range(start, limit, delta) == [3.0, 2.5, 2.0, 1.5]
    * }}}
    *
    * @param  start Start of the number sequence.
    * @param  limit End (exclusive) of the number sequence.
    * @param  delta Difference between consecutive numbers in the sequence.
    * @param  name  Name for the created op.
    * @return Created op output.
    * TODO restrict input and output to (FLOAT32, FLOAT64, INT32, INT64) at compile time
    */
  def range[T <: RealNumericDataType](
    start: Output[RealNumericDataType],
    limit: Output[RealNumericDataType],
    delta: Output[RealNumericDataType] = Basic.constant(1)(INT32),
    name: String = "Range")(
    dataType: T = Set(start.dataType, limit.dataType, delta.dataType).maxBy(_.priority)): Output[T] = {
    var castedStart: Output[DataType] = start
    var castedLimit: Output[DataType] = limit
    var castedDelta: Output[DataType] = delta
    Op.createWith(nameScope = name) {
      val supportedDataTypes = Set[DataType](FLOAT32, FLOAT64, INT32, INT64)
      require(supportedDataTypes.contains(start.dataType), s"Unsupported data type '${start.dataType}'.")
      require(supportedDataTypes.contains(limit.dataType), s"Unsupported data type '${limit.dataType}'.")
      require(supportedDataTypes.contains(delta.dataType), s"Unsupported data type '${delta.dataType}'.")
      if (start.dataType != dataType)
        castedStart = cast(start, dataType)
      if (limit.dataType != dataType)
        castedLimit = cast(limit, dataType)
      if (delta.dataType != dataType)
        castedDelta = cast(delta, dataType)
    }
    Op.Builder(opType = "Range", name = name)
        .addInput(castedStart)
        .addInput(castedLimit)
        .addInput(castedDelta)
        .setAttribute("Tidx", dataType)
        .build().outputs(0).asOutput[T]
  }

  // TODO: Add the "linspace" op.

  /** Creates an op that casts a tensor to a new data type.
    *
    * The op casts `x` to the provided data type.
    *
    * For example:
    * {{{
    *   // `a` is a tensor with values [1.8, 2.2], and data type Float32
    *   cast(a, Int32) == [1, 2] // with data type Int32
    * }}}
    *
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def cast[T <: DataType, U <: DataType](x: Output[T], dataType: U, name: String = "Cast"): Output[U] = {
    Op.Builder(opType = "Cast", name = name)
        .addInput(x)
        .setAttribute("DstT", dataType)
        .build().outputs(0).asOutput[U]
  }

  /** Creates an op that casts a sparse tensor to a new data type.
    *
    * The op casts `x.values` to the provided data type.
    *
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def sparseCast[T <: DataType](x: SparseOutput[DataType], dataType: T, name: String = "Cast"): SparseOutput[T] = {
    val castedValues = Op.Builder(opType = "Cast", name = name)
        .addInput(x.values)
        .setAttribute("DstT", dataType)
        .build().outputs(0).asOutput[T]
    SparseOutput[T](x.indices, castedValues, x.denseShape)
  }

  /** Creates an op that bitcasts a tensor from one type to another without copying data.
    *
    * Given a tensor `input`, the op returns a tensor that has the same buffer data as `input`, but with data type
    * `dataType`. If the input data type `T` is larger (in terms of number of bytes), then the output data type
    * `dataType`, then the shape changes from `[...]` to `[..., sizeof(T)/sizeof(dataType)]`. If `T` is smaller than
    * `dataType`, then the op requires that the rightmost dimension be equal to `sizeof(dataType)/sizeof(T)`. The
    * shape then changes from `[..., sizeof(type)/sizeof(T)]` to `[...]`.
    *
    * *NOTE*: Bitcast is implemented as a low-level cast, so machines with different endian orderings will give
    * different results.
    *
    * @param  input    Input tensor.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def bitcast[T <: DataType](input: Output[T], dataType: DataType, name: String = "Bitcast"): Output[T] = {
    Op.Builder(opType = "Bitcast", name = name)
        .addInput(input)
        .setAttribute("type", dataType)
        .build().outputs(0).asOutput[T]
  }

  @throws[IllegalArgumentException]
  def conjugate[T <: DataType](input: Output[T], name: String = "Conjugate"): Output[T] = {
    if (input.dataType.isComplex) {
      Op.Builder(opType = "Conj", name = name)
          .addInput(input)
          .build().outputs(0).asOutput[T]
    } else if (input.dataType.isNumeric) {
      input
    } else {
      throw new IllegalArgumentException("'conjugate' can only take numeric tensors as input.")
    }
  }

  def addN[T <: DataType](inputs: Array[Output[T]], name: String = "AddN"): Output[T] =
    Op.Builder(opType = "AddN", name = name)
        .addInputList(inputs)
        .build().outputs(0).asOutput[T]

  def matMul[T <: DataType](
      a: Output[T], b: Output[T], transposeA: Boolean = false, transposeB: Boolean = false,
      name: String = "MatMul"): Output[T] = {
    Op.Builder(opType = "MatMul", name = name)
        .addInput(a)
        .addInput(b)
        .setAttribute("transpose_a", transposeA)
        .setAttribute("transpose_b", transposeB)
        .build().outputs(0).asOutput[T]
  }

  def batchMatMul[T <: DataType](
    x: Output[T], y: Output[T], adjointX: Boolean = false, adjointY: Boolean = false,
      name: String = "BatchMatMul"): Output[DataType] =
    Op.Builder(opType = "BatchMatMul", name = name)
        .addInput(x)
        .addInput(y)
        .setAttribute("adj_x", adjointX)
        .setAttribute("adj_y", adjointY)
        .build().outputs(0).asOutput[T]

  //region Unary Ops

  /**
    * Computes numerical negative value element-wise.
    *
    * I.e., \(y = -x\).
    *
    *
    * @param x An [[Output]] or [[SparseOutput]]. Must be one of the following types:
    *          `half`,`float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    *          TODO restrict more precisely to those types (no Qtypes)
    *          TODO sparse
    * @param name A name for the operation (optional).
    * @tparam T The DataType of the operation
    * @return A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
    */
  def negate[T <: DataType](x: Output[T], name: String = "Negate"): Output[T] = {
    Op.Builder(opType = "Neg", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]
  }


  def abs[T <: NumericDataType](x: Output[T], name: String = "Abs"): Output[T] =
    Op.Builder(opType = "Abs", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def complexAbs[T <: DataType](x: Output[T], name: String = "ComplexAbs"): Output[T] =
    Op.Builder(opType = "ComplexAbs", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def reciprocal[T <: DataType](x: Output[T], name: String = "Reciprocal"): Output[T] =
    Op.Builder(opType = "Reciprocal", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def square[T <: DataType](x: Output[T], name: String = "Square"): Output[T] =
    Op.Builder(opType = "Square", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def sqrt[T <: DataType](x: Output[T], name: String = "Sqrt"): Output[T] =
    Op.Builder(opType = "Sqrt", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def reciprocalSqrt[T <: DataType](x: Output[T], name: String = "Rsqrt"): Output[T] =
    Op.Builder(opType = "Rsqrt", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def round[T <: DataType](x: Output[T], name: String = "Round"): Output[T] =
    Op.Builder(opType = "Round", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def exp[T <: DataType](x: Output[T], name: String = "Exp"): Output[T] =
    Op.Builder(opType = "Exp", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def expMinus1[T <: DataType](x: Output[T], name: String = "Expm1"): Output[T] =
    Op.Builder(opType = "Expm1", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def log[T <: DataType](x: Output[T], name: String = "Log"): Output[T] =
    Op.Builder(opType = "Log", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def log1Plus[T <: DataType](x: Output[T], name: String = "Log1p"): Output[T] =
    Op.Builder(opType = "Log1p", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def tanh[T <: DataType](x: Output[T], name: String = "Tanh"): Output[T] =
    Op.Builder(opType = "Tanh", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def logGamma[T <: DataType](x: Output[T], name: String = "Lgamma"): Output[T] =
    Op.Builder(opType = "Lgamma", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def digamma[T <: DataType](x: Output[T], name: String = "Digamma"): Output[T] =
    Op.Builder(opType = "Digamma", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def erf[T <: DataType](x: Output[T], name: String = "Erf"): Output[T] =
    Op.Builder(opType = "Erf", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def complementaryErf[T <: DataType](x: Output[T], name: String = "Erfc"): Output[T] =
    Op.Builder(opType = "Erfc", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def sigmoid[T <: DataType](x: Output[T], name: String = "Sigmoid"): Output[T] =
    Op.Builder(opType = "Sigmoid", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def sin[T <: DataType](x: Output[T], name: String = "Sin"): Output[T] =
    Op.Builder(opType = "Sin", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def cos[T <: DataType](x: Output[T], name: String = "Cos"): Output[T] =
    Op.Builder(opType = "Cos", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def tan[T <: DataType](x: Output[T], name: String = "Tan"): Output[T] =
    Op.Builder(opType = "Tan", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def asin[T <: DataType](x: Output[T], name: String = "Asin"): Output[T] =
    Op.Builder(opType = "Asin", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def acos[T <: DataType](x: Output[T], name: String = "Acos"): Output[T] =
    Op.Builder(opType = "Acos", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def atan[T <: DataType](x: Output[T], name: String = "Atan"): Output[T] =
    Op.Builder(opType = "Atan", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def isNaN[T <: DataType](x: Output[T], name: String = "IsNan"): Output[T] =
    Op.Builder(opType = "IsNan", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def isInf[T <: DataType](x: Output[T], name: String = "IsInf"): Output[T] =
    Op.Builder(opType = "IsInf", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def isFinite[T <: DataType](x: Output[T], name: String = "IsFinite"): Output[T] =
    Op.Builder(opType = "IsFinite", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def sign[T <: DataType](x: Output[T], name: String = "Sign"): Output[T] =
    Op.Builder(opType = "Sign", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def floor[T <: DataType](x: Output[T], name: String = "Floor"): Output[T] =
    Op.Builder(opType = "Floor", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def ceil[T <: DataType](x: Output[T], name: String = "Ceil"): Output[T] =
    Op.Builder(opType = "Ceil", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  def roundInt[T <: DataType](x: Output[T], name: String = "Rint"): Output[T] =
    Op.Builder(opType = "Rint", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[T]

  //endregion Unary Ops

  //region Binary Ops

  def add[T <: DataType](x: Output[T], y: Output[T], name: String = "Add"): Output[T] =
    Op.Builder(opType = "Add", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def subtract[T <: DataType](x: Output[T], y: Output[T], name: String = "Sub"): Output[T] =
    Op.Builder(opType = "Sub", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def multiply[T <: DataType](x: Output[T], y: Output[T], name: String = "Mul"): Output[T] =
    Op.Builder(opType = "Mul", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def divide[T <: DataType](x: Output[T], y: Output[T], name: String = "Div"): Output[T] =
    Op.Builder(opType = "Div", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def floorDivide[T <: DataType](x: Output[T], y: Output[T], name: String = "FloorDiv"): Output[T] =
    Op.Builder(opType = "FloorDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def truncateDivide[T <: DataType](x: Output[T], y: Output[T], name: String = "TruncateDiv"): Output[T] =
    Op.Builder(opType = "TruncateDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def realDivide[T <: DataType](x: Output[T], y: Output[T], name: String = "RealDiv"): Output[T] =
    Op.Builder(opType = "RealDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def squaredDifference[T <: DataType](x: Output[T], y: Output[T], name: String = "SquaredDifference"): Output[T] =
    Op.Builder(opType = "SquaredDifference", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def maximum[T <: DataType](x: Output[T], y: Output[T], name: String = "Maximum"): Output[T] =
    Op.Builder(opType = "Maximum", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def minimum[T <: DataType](x: Output[T], y: Output[T], name: String = "Minimum"): Output[T] =
    Op.Builder(opType = "Minimum", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def mod[T <: DataType](x: Output[T], y: Output[T], name: String = "Mod"): Output[T] =
    Op.Builder(opType = "Mod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def floorMod[T <: NumericDataType](x: Output[T], y: Output[T], name: String = "FloorMod"): Output[T] =
    Op.Builder(opType = "FloorMod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def truncateMod[T <: NumericDataType](x: Output[T], y: Output[T], name: String = "TruncateMod"): Output[T] =
    Op.Builder(opType = "TruncateMod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def pow[T <: DataType, U <: NumericDataType](x: Output[T], y: Output[U], name: String = "Pow"): Output[T] =
    Op.Builder(opType = "Pow", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]

  def igammac(a: Output[DataType], x: Output[DataType], name: String = "Igammac"): Output[DataType] =
    Op.Builder(opType = "Igammac", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)

  def igamma(a: Output[DataType], x: Output[DataType], name: String = "Igamma"): Output[DataType] =
    Op.Builder(opType = "Igamma", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)

  def zeta[T <: DataType](x: Output[T], q: Output[DataType], name: String = "Zeta"): Output[DataType] =
    Op.Builder(opType = "Zeta", name = name)
        .addInput(x)
        .addInput(q)
        .build().outputs(0)

  def polygamma(a: Output[DataType], x: Output[DataType], name: String = "Polygamma"): Output[DataType] =
    Op.Builder(opType = "Polygamma", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)

  //endregion Binary Ops

  def betainc(a: Output[DataType], b: Output[DataType], x: Output[DataType], name: String = "Betainc"): Output[DataType] =
    Op.Builder(opType = "Betainc", name = name)
        .addInput(a)
        .addInput(b)
        .addInput(x)
        .build().outputs(0)

  //region Logical Ops

  /** Creates an op that computes the truth value of `!x` element-wise.
    *
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalNot[T <: DataType](x: Output[T], name: String = "LogicalNot")(implicit ev: T <:< BOOLEAN): Output[BOOLEAN] = {
    Op.Builder(opType = "LogicalNot", name = name)
        .addInput(x)
        .build().outputs(0).asOutput[BOOLEAN]
  }

  /** Creates an op that computes the truth value of `x && y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalAnd[T <: DataType](x: Output[T], y: Output[T], name: String = "LogicalAnd")(implicit ev: T <:< BOOLEAN): Output[BOOLEAN] = {
    Op.Builder(opType = "LogicalAnd", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[BOOLEAN]
  }

  /** Creates an op that computes the truth value of `x || y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalOr[T <: DataType](x: Output[T], y: Output[T], name: String = "LogicalOr")(implicit ev: T <:< BOOLEAN): Output[BOOLEAN] = {
    Op.Builder(opType = "LogicalOr", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[BOOLEAN]
  }

  /** Creates an op that computes the truth value of `(x || y) && !(x && y)` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalXor[T <: DataType](x: Output[T], y: Output[T], name: String = "LogicalXor")(implicit ev: T <:< BOOLEAN): Output[BOOLEAN] = {
    logicalAnd(logicalOr(x, y), logicalNot(logicalAnd(x, y)), name = name)
  }

  //endregion Logical Ops

  //region Comparison Ops

  /** Creates an op that computes the truth value of `x == y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def equal[T <: DataType](x: Output[T], y: Output[T], name: String = "Equal"): Output[BOOLEAN] = {
    Op.Builder(opType = "Equal", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[BOOLEAN]
  }

  /** Creates an op that computes the truth value of `x != y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def notEqual[T <: DataType](x: Output[T], y: Output[T], name: String = "NotEqual"): Output[BOOLEAN] = {
    Op.Builder(opType = "NotEqual", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[BOOLEAN]
  }

  /** Creates an op that computes the truth value of `abs(x - y) < tolerance`  element-wise.
    *
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  tolerance Comparison tolerance value.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def approximatelyEqual[T <: DataType](
      x: Output[T], y: Output[T], tolerance: Float = 0.00001f, name: String = "ApproximatelyEqual"): Output[T] = {
    Op.Builder(opType = "ApproximateEqual", name = name)
        .addInput(x)
        .addInput(y)
        .setAttribute("tolerance", tolerance)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the truth value of `x < y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def less[T <: DataType](x: Output[T], y: Output[T], name: String = "Less"): Output[T] = {
    Op.Builder(opType = "Less", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the truth value of `x <= y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def lessEqual[T <: DataType](x: Output[T], y: Output[T], name: String = "LessEqual"): Output[T] = {
    Op.Builder(opType = "LessEqual", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the truth value of `x > y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def greater[T <: DataType](x: Output[T], y: Output[T], name: String = "Greater"): Output[T] = {
    Op.Builder(opType = "Greater", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the truth value of `x >= y` element-wise.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def greaterEqual[T <: DataType](x: Output[T], y: Output[T], name: String = "GreaterEqual"): Output[T] = {
    Op.Builder(opType = "GreaterEqual", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0).asOutput[T]
  }

  //endregion Comparison Ops

  //region Reduction Ops

  private[this] def reductionAxes(tensor: Output[DataType], axes: Output[DataType]): Output[DataType] = {
    if (axes != null)
      axes
    else
      Basic.constant(Tensor.fromSeq(0 until tensor.shape.rank: _*)(INT32.supportedType))()
  }

  /** Creates an op that computes the sum of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1, 1, 1]], [1, 1, 1]]
    *   sum(x) == 6
    *   sum(x, 0) == [2, 2, 2]
    *   sum(x, 1) == [3, 3]
    *   sum(x, 1, keepDims = true) == [[3], [3]]
    *   sum(x, [0, 1]) == 6
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    * TODO restrict axes to INT32 | INT64
    */
  def sum[T <: DataType, U <: RealNumericDataType](
    input: Output[T], axes: Output[U] = null, keepDims: Boolean = false, name: String = "Sum"): Output[T] = {
    Op.Builder(opType = "Sum", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the mean of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *   mean(x) == 1.5
    *   mean(x, 0) == [1.5, 1.5]
    *   mean(x, 1) == [1.0, 2.0]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def mean[T <: DataType](input: Output[T], axes: Output[DataType] = null, keepDims: Boolean = false, name: String = "Mean"): Output[T] = {
    Op.Builder(opType = "Mean", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the product of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1, 1, 1]], [1, 1, 1]]
    *   product(x) == 1
    *   product(x, 0) == [1, 1, 1]
    *   product(x, 1) == [1, 1]
    *   product(x, 1, keepDims = true) == [[1], [1]]
    *   product(x, [0, 1]) == 1
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def product[T <: DataType](input: Output[T], axes: Output[DataType] = null, keepDims: Boolean = false, name: String = "Product"): Output[T] = {
    Op.Builder(opType = "Prod", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the minimum of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *   min(x) == 1.0
    *   min(x, 0) == [1.0, 1.0]
    *   min(x, 1) == [1.0, 2.0]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    * TODO restrict axis to INT32|INT64
    */
  def min[T <: NumericDataType, U <: RealNumericDataType](input: Output[T], axes: Output[U] = null, keepDims: Boolean = false, name: String = "Min"): Output[T] = {
    Op.Builder(opType = "Min", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the maximum of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *   max(x) == 2.0
    *   max(x, 0) == [2.0, 2.0]
    *   max(x, 1) == [1.0, 2.0]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    * TODO restrict axes to INT32|INT64
    */
  def max[T <: NumericDataType, U <: RealNumericDataType](input: Output[T], axes: Output[U] = null, keepDims: Boolean = false, name: String = "Max"): Output[T] = {
    Op.Builder(opType = "Max", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the logical AND of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[true, true], [false, false]]
    *   all(x) == false
    *   all(x, 0) == [false, false]
    *   all(x, 1) == [true, false]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    * TODO restrict axes to INT32|INT64
    */
  def all[T <: RealNumericDataType](input: Output[BOOLEAN], axes: Output[T] = null, keepDims: Boolean = false, name: String = "All"): Output[BOOLEAN] = {
    Op.Builder(opType = "All", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0).asOutput[BOOLEAN]
  }

  /** Creates an op that computes the logical OR of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[true, true], [false, false]]
    *   any(x) == true
    *   any(x, 0) == [true, true]
    *   any(x, 1) == [true, false]
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    * TODO restrict axes to INT32|INT64
    */
  def any[T <: RealNumericDataType](
    input: Output[BOOLEAN], axes: Output[T] = null, keepDims: Boolean = false, name: String = "Any"): Output[BOOLEAN] = {
      Op.Builder(opType = "Any", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0).asOutput[BOOLEAN]
  }

  /** Creates an op that computes the log-sum-exp of elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * This function is more numerically stable than `log(sum(exp(input)))`. It avoids overflows caused by computing the
    * exponential of large inputs, and underflows caused by computing the logarithm of small inputs.
    *
    * For example:
    * {{{
    *   // 'x' is [[0, 0, 0], [0, 0, 0]]
    *   logSumExp(x) == log(6)
    *   logSumExp(x, 0) == [log(2), log(2), log(2)]
    *   logSumExp(x, 1) == [log(3), log(3)]
    *   logSumExp(x, 1, keepDims = true) == [[log(3)], [log(3)]]
    *   logSumExp(x, [0, 1]) == log(6)
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def logSumExp[T <: NumericDataType](
      input: Output[T], axes: Array[Int] = null, keepDims: Boolean = false, name: String = "LogSumExp"): Output[T] = {
    // TODO: !!! Can we support a dynamic version for the axes argument?
    Op.createWith(nameScope = name) {
      val maxValue = Basic.stopGradient(max[T, INT32](input, axes, keepDims = true))
      val result = log(sum(exp(input - maxValue), axes, keepDims = true)) + maxValue
      if (keepDims)
        result
      else
        Basic.squeeze(result, axes)
    }
  }

  /** Creates an op that computes the number of non-zero elements across dimensions of a tensor.
    *
    * Reduces `input` along the dimensions given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is
    * reduced by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axis` is `null`, then all dimensions are reduced, and a tensor with a single element is returned.
    *
    * IMPORTANT NOTE: Floating point comparison to zero is done by exact floating point equality check. Small values are
    * **not** rounded to zero for the purposes of the non-zero check.
    *
    * For example:
    * {{{
    *   // 'x' is [[0, 1, 0], [1, 1, 0]]
    *   countNonZero(x) == 3
    *   countNonZero(x, 0) == [1, 2, 0]
    *   countNonZero(x, 1) == [1, 2]
    *   countNonZero(x, 1, keepDims = true) == [[1], [2]]
    *   countNonZero(x, [0, 1]) == 3
    * TODO restrict axes to INT32|INT64
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output with `INT64` data type.
    */
  def countNonZero[T <: DataType, U <: RealNumericDataType](
      input: Output[T], axes: Output[U] = null, keepDims: Boolean = false, name: String = "CountNonZero"): Output[INT64] = {
    Op.createWith(nameScope = name) {
      sum(cast(notEqual(input, Basic.constant(0)()), INT64), axes, keepDims)
    }
  }

  //endregion Reduction Ops

  //region Segment Ops

  /** Creates an op that computes the sum along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \sum_{j...} data(j,...)` where the sum is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `unsortedSegmentSum`, `segmentIndices` need be sorted.
    *
    * If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `INT32` or `INT64`). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentSum[T <: NumericDataType, U <: NumericDataType](
    data: Output[T], segmentIndices: Output[T], name: String = "SegmentSum"): Output[T] = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentSum", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the mean along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \frac{sum_{j...} data(j,...)}{N}` where the sum is over all `j`
    * such that `segmentIndices(j) == i` and `N` is the total number of values being summed. Unlike
    * `unsortedSegmentMean`, `segmentIndices` need be sorted.
    *
    * If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `INT32` or `INT64`). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentMean[T <: NumericDataType, U <: NumericDataType](
    data: Output[T], segmentIndices: Output[U], name: String = "SegmentMean"): Output[T] = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentMean", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the product along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \prod_{j...} data(j,...)` where the product is over all `j` such
    * that `segmentIndices(j) == i`. Unlike `unsortedSegmentProd`, `segmentIndices` need be sorted.
    *
    * If the product if empty for a given segment index `i`, `output(i)` is set to `1`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `INT32` or `INT64`). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentProd[T <: NumericDataType, U <: NumericDataType](
    data: Output[T], segmentIndices: Output[U], name: String = "SegmentProd"): Output[T] = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentProd", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the min along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \min_{j...} data(j,...)` where the min is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `unsortedSegmentMin`, `segmentIndices` need be sorted.
    *
    * If the min if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `INT32` or `INT64`). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentMin[T <: NumericDataType, U <: NumericDataType](
    data: Output[T], segmentIndices: Output[U], name: String = "SegmentMin"): Output[T] = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentMin", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the max along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \max_{j...} data(j,...)` where the max is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `unsortedSegmentMax`, `segmentIndices` need be sorted.
    *
    * If the max if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `INT32` or `INT64`). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def segmentMax[T <: NumericDataType, U <: NumericDataType](
    data: Output[T], segmentIndices: Output[U], name: String = "SegmentMax"): Output[T] = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentMax", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the sum along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \sum_{j...} data(j...)` where the sum is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `segmentSum`, `segmentIndices` need not be sorted and need not cover all values
    * in the full range of valid values.
    *
    * If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * `segmentsNumber` should equal the number of distinct segment indices.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `INT32` or `INT64`).
    * @param  segmentsNumber Number of segments (must have data type of `INT32`).
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def unsortedSegmentSum[T <: NumericDataType, U <: RealNumericDataType](
      data: Output[T], segmentIndices: Output[U], segmentsNumber: Output[INT32], name: String = "UnsortedSegmentSum"): Output[T] = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    if (segmentsNumber.dataType != INT32)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentsNumber.dataType}', is not 'INT32', as required.")
    Op.Builder(opType = "UnsortedSegmentSum", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .addInput(segmentsNumber)
        .build().outputs(0).asOutput[T]
  }

  /** Creates an op that computes the max along segments of a tensor.
    *
    * The op computes a tensor such that `output(i) = \max_{j...} data(j...)` where the max is over all `j` such that
    * `segmentIndices(j) == i`. Unlike `segmentMax`, `segmentIndices` need not be sorted and need not cover all values
    * in the full range of valid values.
    *
    * If the max if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    * `segmentsNumber` should equal the number of distinct segment indices.
    *
    * The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    * distinct segment indices.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of `INT32` or `INT64`).
    * @param  segmentsNumber Number of segments (must have data type of `INT32`).
    * @param  name           Name for the created op.
    * @return Created op.
    */
  def unsortedSegmentProd[T <: NumericDataType, U <: NumericDataType, V <: NumericDataType](
      data: Output[T], segmentIndices: Output[T], segmentsNumber: Output[T], name: String = "UnsortedSegmentMax"): Output[T] = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    if (segmentsNumber.dataType != INT32)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentsNumber.dataType}', is not 'INT32', as required.")
    Op.Builder(opType = "UnsortedSegmentMax", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .addInput(segmentsNumber)
        .build().outputs(0).asOutput[T]
  }

  // TODO: [SPARSE] Add sparse segment ops.

  //endregion Segment Ops
}

object Math extends Math {
  private[api] object Gradients {
    GradientsRegistry.register("Diag", diagGradient)
    GradientsRegistry.register("DiagPart", diagPartGradient)
    GradientsRegistry.register("MatrixDiag", matrixDiagGradient)
    GradientsRegistry.register("MatrixSetDiag", matrixSetDiagGradient)
    GradientsRegistry.register("MatrixDiagPart", matrixDiagPartGradient)
    GradientsRegistry.register("MatrixBandPart", matrixBandPartGradient)
    GradientsRegistry.register("Cast", castGradient)
    GradientsRegistry.register("MatMul", matMulGradient)
    GradientsRegistry.register("BatchMatMul", batchMatMulGradient)
    GradientsRegistry.register("Square", squareGradient)
    GradientsRegistry.register("Sub", subtractGradient)
    GradientsRegistry.register("Sum", reduceSumGradient)

    private[this] def diagGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      Seq(diagPart(outputGradients.head))
    }

    private[this] def diagPartGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      Seq(diag(outputGradients.head))
    }

    private[this] def matrixDiagGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      Seq(matrixDiagPart(outputGradients.head))
    }

    private[this] def matrixSetDiagGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      val gradient = outputGradients.head
      val inputShape = op.inputs(0).shape.mergeWith(gradient.shape)
      val batchShape = inputShape(0 :: -2).mergeWith(op.inputs(1).shape(0 :: -1))
      val matrixShape = inputShape(-2 ::)
      val diagShape = {
        if (batchShape.isFullyDefined && matrixShape.isFullyDefined) {
          Basic.constant(Tensor((batchShape.asArray :+ matrixShape.asArray.min).map(Tensor(_)): _*))(INT32)
        } else {
          Op.colocateWith(Set(gradient.op)) {
            val gradShape = Basic.shape[DataType, INT32](gradient)
            val gradRank = Basic.rank(gradient)
            val batchShape = Basic.slice(gradShape, 0, gradRank - 2)
            val matrixShape = Basic.slice(gradShape, gradRank - 2, 2)
            val minDim = min[INT32, INT32](matrixShape)
            Basic.concatenate(Seq(batchShape, minDim), 0)
          }
        }
      }
      val gradInput = matrixSetDiag(gradient, Basic.fill(diagShape, Tensor(gradient.dataType, 0)))
      val gradDiag = matrixDiagPart(gradient)
      Seq(gradInput, gradDiag)
    }

    private[this] def matrixDiagPartGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      val matrixShape = op.inputs(0).shape(-2 ::)
      if (matrixShape.isFullyDefined && matrixShape(0) == matrixShape(1))
        Seq(matrixDiag(outputGradients.head))
      else {
        Seq(matrixSetDiag(Basic.zerosLike(op.inputs(0))(), outputGradients.head))
      }
    }

    private[this] def matrixBandPartGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      Seq(matrixBandPart(outputGradients.head, op.inputs(1), op.inputs(2)), null, null)
    }

    // TODO restrict to FLOAT32|FLOAT64 at compile time
    private[this] def castGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      val supportedDataTypes = Seq(FLOAT32, FLOAT64) // TODO: [TYPES] Float16 and complex.
      val sourceDataType = op.inputs(0).dataType
      val destinationDataType = outputGradients.head.dataType
      if (supportedDataTypes.contains(sourceDataType) && supportedDataTypes.contains(destinationDataType))
        Seq(cast(outputGradients.head, sourceDataType))
      else
        Seq(null)
    }

    private[this] def matMulGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      matMulGradientCommon(op, outputGradients, "transpose_a", "transpose_b", isBatch = false)
    }

    private[this] def batchMatMulGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      matMulGradientCommon(op, outputGradients, "adj_x", "adj_y", isBatch = true)
    }

    private[this] def matMulGradientCommon(
        op: Op, outputGradients: Seq[OutputLike[DataType]], transposeAAttribute: String, transposeBAttribute: String,
        isBatch: Boolean): Seq[OutputLike[DataType]] = {
      val transposeA = op.booleanAttribute(transposeAAttribute)
      val transposeB = op.booleanAttribute(transposeBAttribute)
      val a = conjugate(op.inputs(0))
      val b = conjugate(op.inputs(1))
      val outputGradient = outputGradients.head
      if (!transposeA && !transposeB)
        matMulGradientHelper(
          outputGradient, b, a, outputGradient,
          transposeX0 = false, transposeX1 = true, transposeY0 = true, transposeY1 = false, isBatch = isBatch)
      else if (!transposeA && transposeB)
        matMulGradientHelper(
          outputGradient, b, outputGradient, a,
          transposeX0 = false, transposeX1 = false, transposeY0 = true, transposeY1 = false, isBatch = isBatch)
      else if (transposeA && !transposeB)
        matMulGradientHelper(
          b, outputGradient, a, outputGradient,
          transposeX0 = false, transposeX1 = true, transposeY0 = false, transposeY1 = false, isBatch = isBatch)
      else
        matMulGradientHelper(
          b, outputGradient, outputGradient, a,
          transposeX0 = true, transposeX1 = true, transposeY0 = true, transposeY1 = true, isBatch = isBatch)
    }

    private[this] def matMulGradientHelper(
        x0: Output[DataType], x1: Output[DataType], y0: Output[DataType], y1: Output[DataType], transposeX0: Boolean, transposeX1: Boolean,
        transposeY0: Boolean, transposeY1: Boolean, isBatch: Boolean): Seq[OutputLike[DataType]] = {
      if (!isBatch) {
        val gradientX = matMul(x0, x1, transposeA = transposeX0, transposeB = transposeX1, name = "MatMul_1")
        val gradientY = matMul(y0, y1, transposeA = transposeY0, transposeB = transposeY1, name = "MatMul_2")
        Seq[OutputLike[DataType]](gradientX, gradientY)
      } else {
        val gradientX = batchMatMul(x0, x1, adjointX = transposeX0, adjointY = transposeX1, name = "MatMul_1")
        val gradientY = batchMatMul(y0, y1, adjointX = transposeY0, adjointY = transposeY1, name = "MatMul_2")
        Seq[OutputLike[DataType]](gradientX, gradientY)
      }
    }

    private[this] def squareGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      var x = op.inputs(0)
      val outputGradient = outputGradients.head
      // Using control dependencies to prevent 2*x from being computed too early.
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        x = conjugate(x)
        // TODO: !!! Automatic casting for mathematic operations? At least print nicely formatted exception?
        Seq(outputGradient * (Basic.constant(2)(x.dataType) * x))
      }
    }

    /** Returns the reduction indices for computing the gradients of `shape0` `[operator]` `shape1` with broadcasting.
      *
      * This is typically used by gradient computations for broadcasting operations.
      *
      * @param  shape0 First operand shape.
      * @param  shape1 Second operand shape.
      * @param  name   Name for the created op.
      * @return Tuple containing two op outputs, each containing the reduction indices for the corresponding op.
      */
    private[this] def broadcastGradientArguments[T <: DataType](
        shape0: Output[T], shape1: Output[T], name: String = "BroadcastGradientArguments"): (Output[T], Output[T]) = {
      val outputs = Op.Builder(opType = "BroadcastGradientArgs", name = name)
          .addInput(shape0)
          .addInput(shape1)
          .build().outputs
      (outputs(0).asOutput[T], outputs(1).asOutput[T])
    }

    private[this] def subtractGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      val x = op.inputs(0)
      val y = op.inputs(1)
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val (rx, ry) = broadcastGradientArguments(xShape, yShape)
      val outputGradient = outputGradients.head
      val gradientX = Basic.reshape(sum(outputGradient, rx), xShape)
      val gradientY = -Basic.reshape(sum(outputGradient, ry), yShape)
      Seq(gradientX, gradientY)
    }

    private[this] def reduceSumGradient(op: Op, outputGradients: Seq[OutputLike[DataType]]): Seq[OutputLike[DataType]] = {
      // Fast path for when reducing to a scalar and rank is known, which adds only reshape and tile ops (and possibly a
      // shape op too).
      if (op.inputs(0).shape.rank != -1 && op.inputs(1).op.opType == "Const") {
        val rank = op.inputs(0).shape.rank
        // TODO: !!! Missing a pretty important if statement here.
        val gradient = Basic.reshape(outputGradients.head, Shape.fromSeq(Seq.fill(rank)(1)))
        // If shape is not fully defined (but rank is), we use a shape op.
        if (op.inputs(0).shape.isFullyDefined)
          Seq(Basic.tile(gradient, op.inputs(0).shape), null)
        else
          Seq(Basic.tile(gradient, Basic.shape(op.inputs(0))), null)
      } else {
        ???
      }
    }
  }
}
