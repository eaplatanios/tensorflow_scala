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
import org.platanios.tensorflow.api.types.{DataType, FLOAT32, FLOAT64, INT32, INT64}

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
  def select(condition: Output, x: Output, y: Output, name: String = "Select"): Output = {
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
  def diag(diagonal: Output, name: String = "Diag"): Output = {
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
  def diagPart(input: Output, name: String = "DiagPart"): Output = {
    if (input.rank != 2 && input.rank != 4 && input.rank != 6)
      throw new IllegalArgumentException(s"The provided tensor (rank = ${input.rank}) can only be 2, 4, or 6.")
    Op.Builder(opType = "DiagPart", name = name)
        .addInput(input)
        .build().outputs(0)
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
  def matrixDiag(diagonal: Output, name: String = "MatrixDiag"): Output = {
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
  def matrixSetDiag(input: Output, diagonal: Output, name: String = "MatrixSetDiag"): Output = {
    if (input.rank > -1 && input.rank < 2)
      throw new IllegalArgumentException(s"The provided input tensor (rank = ${input.rank}) must have rank at least 2.")
    if (diagonal.rank > -1 && diagonal.rank < 1)
      throw new IllegalArgumentException(
        s"The provided diagonal tensor (rank = ${diagonal.rank}) must have rank at least 1.")
    Op.Builder(opType = "MatrixSetDiag", name = name)
        .addInput(input)
        .addInput(diagonal)
        .build().outputs(0)
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
  def matrixDiagPart(input: Output, name: String = "MatrixDiagPart"): Output = {
    if (input.rank > -1 && input.rank < 2)
      throw new IllegalArgumentException(s"The provided input tensor (rank = ${input.rank}) must have rank at least 2.")
    Op.Builder(opType = "MatrixDiagPart", name = name)
        .addInput(input)
        .build().outputs(0)
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
  def matrixBandPart(
      input: Output, numSubDiagonals: Output, numSuperDiagonals: Output, name: String = "MatrixBandPart"): Output = {
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
        .build().outputs(0)
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
    */
  def range(
      start: Output, limit: Output, delta: Output = Basic.constant(1), dataType: DataType = null,
      name: String = "Range"): Output = {
    var castedStart: Output = null
    var castedLimit: Output = null
    var castedDelta: Output = null
    Op.createWith(nameScope = name) {
      val supportedDataTypes = Set[DataType](FLOAT32, FLOAT64, INT32, INT64)
      require(supportedDataTypes.contains(start.dataType), s"Unsupported data type '${start.dataType}'.")
      require(supportedDataTypes.contains(limit.dataType), s"Unsupported data type '${limit.dataType}'.")
      require(supportedDataTypes.contains(delta.dataType), s"Unsupported data type '${delta.dataType}'.")
      val inferredDataType = {
        if (dataType != null)
          dataType
        else
          Set(start.dataType, limit.dataType, delta.dataType).maxBy(_.priority)
      }
      if (start.dataType != inferredDataType)
        castedStart = cast(start, inferredDataType)
      if (limit.dataType != inferredDataType)
        castedLimit = cast(limit, inferredDataType)
      if (delta.dataType != inferredDataType)
        castedDelta = cast(delta, inferredDataType)
    }
    Op.Builder(opType = "Range", name = name)
        .addInput(castedStart)
        .addInput(castedLimit)
        .addInput(castedDelta)
        .build().outputs(0)
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
    * **NOTE**: Only a smaller number of types are supported by the `cast` op. The exact casting rule is TBD. The
    * current implementation uses C++ static cast rules for numeric types, which may be changed in the future.
    *
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def cast(x: Output, dataType: DataType, name: String = "Cast"): Output = {
    Op.Builder(opType = "Cast", name = name)
        .addInput(x)
        .setAttribute("DstT", dataType)
        .build().outputs(0)
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
  def sparseCast(x: SparseOutput, dataType: DataType, name: String = "Cast"): SparseOutput = {
    val castedValues = Op.Builder(opType = "Cast", name = name)
        .addInput(x.values)
        .setAttribute("DstT", dataType)
        .build().outputs(0)
    SparseOutput(x.indices, castedValues, x.denseShape)
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
  def bitcast(input: Output, dataType: DataType, name: String = "Bitcast"): Output = {
    Op.Builder(opType = "Bitcast", name = name)
        .addInput(input)
        .setAttribute("type", dataType)
        .build().outputs(0)
  }

  @throws[IllegalArgumentException]
  def conjugate(input: Output, name: String = "Conjugate"): Output = {
    if (input.dataType.isComplex) {
      Op.Builder(opType = "Conj", name = name)
          .addInput(input)
          .build().outputs(0)
    } else if (input.dataType.isNumeric) {
      input
    } else {
      throw new IllegalArgumentException("'conjugate' can only take numeric tensors as input.")
    }
  }

  /** Creates an op that adds all input tensors element-wise.
    *
    * @param  inputs Input tensors (must all have the same shape and size).
    * @param  name   Created op name.
    * @return Created op output.
    */
  def addN(inputs: Array[Output], name: String = "AddN"): Output = {
    Op.Builder(opType = "AddN", name = name)
        .addInputList(inputs)
        .build().outputs(0)
  }

  def matMul(
      a: Output, b: Output, transposeA: Boolean = false, transposeB: Boolean = false,
      name: String = "MatMul"): Output = {
    if (a.rank != 2)
      throw new IllegalArgumentException(s"Expected a two-dimensional tensor as input, but 'a' has rank ${a.rank}.")
    if (b.rank != 2)
      throw new IllegalArgumentException(s"Expected a two-dimensional tensor as input, but 'b' has rank ${b.rank}.")
    Op.Builder(opType = "MatMul", name = name)
        .addInput(a)
        .addInput(b)
        .setAttribute("transpose_a", transposeA)
        .setAttribute("transpose_b", transposeB)
        .build().outputs(0)
  }

  def batchMatMul(
      x: Output, y: Output, adjointX: Boolean = false, adjointY: Boolean = false,
      name: String = "BatchMatMul"): Output = {
    Op.Builder(opType = "BatchMatMul", name = name)
        .addInput(x)
        .addInput(y)
        .setAttribute("adj_x", adjointX)
        .setAttribute("adj_y", adjointY)
        .build().outputs(0)
  }

  //region Unary Ops

  /** Creates an op that computes the absolute value of a tensor.
    *
    * Given a tensor `x`, the op returns a tensor containing the absolute value of each element in `x`. For example, if
    * `x` is an input element and `y` is an output element, the op computes `y = |x|`.
    *
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def abs[T: OutputOps](x: T, name: String = "Abs"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o =>
          Op.Builder(opType = "Abs", name = name)
              .addInput(o)
              .build()
              .outputs(0))
  }

  def negate(x: Output, name: String = "Negate"): Output = {
    Op.Builder(opType = "Neg", name = name)
        .addInput(x)
        .build().outputs(0)
  }

  def complexAbs(x: Output, name: String = "ComplexAbs"): Output =
    Op.Builder(opType = "ComplexAbs", name = name)
        .addInput(x)
        .build().outputs(0)

  def reciprocal(x: Output, name: String = "Reciprocal"): Output =
    Op.Builder(opType = "Reciprocal", name = name)
        .addInput(x)
        .build().outputs(0)

  def square(x: Output, name: String = "Square"): Output =
    Op.Builder(opType = "Square", name = name)
        .addInput(x)
        .build().outputs(0)

  def sqrt(x: Output, name: String = "Sqrt"): Output =
    Op.Builder(opType = "Sqrt", name = name)
        .addInput(x)
        .build().outputs(0)

  def reciprocalSqrt(x: Output, name: String = "Rsqrt"): Output =
    Op.Builder(opType = "Rsqrt", name = name)
        .addInput(x)
        .build().outputs(0)

  def round(x: Output, name: String = "Round"): Output =
    Op.Builder(opType = "Round", name = name)
        .addInput(x)
        .build().outputs(0)

  def exp(x: Output, name: String = "Exp"): Output =
    Op.Builder(opType = "Exp", name = name)
        .addInput(x)
        .build().outputs(0)

  def expMinus1(x: Output, name: String = "Expm1"): Output =
    Op.Builder(opType = "Expm1", name = name)
        .addInput(x)
        .build().outputs(0)

  def log(x: Output, name: String = "Log"): Output =
    Op.Builder(opType = "Log", name = name)
        .addInput(x)
        .build().outputs(0)

  def log1Plus(x: Output, name: String = "Log1p"): Output =
    Op.Builder(opType = "Log1p", name = name)
        .addInput(x)
        .build().outputs(0)

  def tanh(x: Output, name: String = "Tanh"): Output =
    Op.Builder(opType = "Tanh", name = name)
        .addInput(x)
        .build().outputs(0)

  def logGamma(x: Output, name: String = "Lgamma"): Output =
    Op.Builder(opType = "Lgamma", name = name)
        .addInput(x)
        .build().outputs(0)

  def digamma(x: Output, name: String = "Digamma"): Output =
    Op.Builder(opType = "Digamma", name = name)
        .addInput(x)
        .build().outputs(0)

  def erf(x: Output, name: String = "Erf"): Output =
    Op.Builder(opType = "Erf", name = name)
        .addInput(x)
        .build().outputs(0)

  def complementaryErf(x: Output, name: String = "Erfc"): Output =
    Op.Builder(opType = "Erfc", name = name)
        .addInput(x)
        .build().outputs(0)

  def sigmoid(x: Output, name: String = "Sigmoid"): Output =
    Op.Builder(opType = "Sigmoid", name = name)
        .addInput(x)
        .build().outputs(0)

  def sin(x: Output, name: String = "Sin"): Output =
    Op.Builder(opType = "Sin", name = name)
        .addInput(x)
        .build().outputs(0)

  def cos(x: Output, name: String = "Cos"): Output =
    Op.Builder(opType = "Cos", name = name)
        .addInput(x)
        .build().outputs(0)

  def tan(x: Output, name: String = "Tan"): Output =
    Op.Builder(opType = "Tan", name = name)
        .addInput(x)
        .build().outputs(0)

  def asin(x: Output, name: String = "Asin"): Output =
    Op.Builder(opType = "Asin", name = name)
        .addInput(x)
        .build().outputs(0)

  def acos(x: Output, name: String = "Acos"): Output =
    Op.Builder(opType = "Acos", name = name)
        .addInput(x)
        .build().outputs(0)

  def atan(x: Output, name: String = "Atan"): Output =
    Op.Builder(opType = "Atan", name = name)
        .addInput(x)
        .build().outputs(0)

  def isNaN(x: Output, name: String = "IsNan"): Output =
    Op.Builder(opType = "IsNan", name = name)
        .addInput(x)
        .build().outputs(0)

  def isInf(x: Output, name: String = "IsInf"): Output =
    Op.Builder(opType = "IsInf", name = name)
        .addInput(x)
        .build().outputs(0)

  def isFinite(x: Output, name: String = "IsFinite"): Output =
    Op.Builder(opType = "IsFinite", name = name)
        .addInput(x)
        .build().outputs(0)

  def sign(x: Output, name: String = "Sign"): Output =
    Op.Builder(opType = "Sign", name = name)
        .addInput(x)
        .build().outputs(0)

  def floor(x: Output, name: String = "Floor"): Output =
    Op.Builder(opType = "Floor", name = name)
        .addInput(x)
        .build().outputs(0)

  def ceil(x: Output, name: String = "Ceil"): Output =
    Op.Builder(opType = "Ceil", name = name)
        .addInput(x)
        .build().outputs(0)

  def roundInt(x: Output, name: String = "Rint"): Output =
    Op.Builder(opType = "Rint", name = name)
        .addInput(x)
        .build().outputs(0)

  //endregion Unary Ops

  //region Binary Ops

  def add(x: Output, y: Output, name: String = "Add"): Output =
    Op.Builder(opType = "Add", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def subtract(x: Output, y: Output, name: String = "Sub"): Output =
    Op.Builder(opType = "Sub", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def multiply(x: Output, y: Output, name: String = "Mul"): Output =
    Op.Builder(opType = "Mul", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def divide(x: Output, y: Output, name: String = "Div"): Output =
    Op.Builder(opType = "Div", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def floorDivide(x: Output, y: Output, name: String = "FloorDiv"): Output =
    Op.Builder(opType = "FloorDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def truncateDivide(x: Output, y: Output, name: String = "TruncateDiv"): Output =
    Op.Builder(opType = "TruncateDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def realDivide(x: Output, y: Output, name: String = "RealDiv"): Output =
    Op.Builder(opType = "RealDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def squaredDifference(x: Output, y: Output, name: String = "SquaredDifference"): Output =
    Op.Builder(opType = "SquaredDifference", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def maximum(x: Output, y: Output, name: String = "Maximum"): Output =
    Op.Builder(opType = "Maximum", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def minimum(x: Output, y: Output, name: String = "Minimum"): Output =
    Op.Builder(opType = "Minimum", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def mod(x: Output, y: Output, name: String = "Mod"): Output =
    Op.Builder(opType = "Mod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def floorMod(x: Output, y: Output, name: String = "FloorMod"): Output =
    Op.Builder(opType = "FloorMod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def truncateMod(x: Output, y: Output, name: String = "TruncateMod"): Output =
    Op.Builder(opType = "TruncateMod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def pow(x: Output, y: Output, name: String = "Pow"): Output =
    Op.Builder(opType = "Pow", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)

  def igammac(a: Output, x: Output, name: String = "Igammac"): Output =
    Op.Builder(opType = "Igammac", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)

  def igamma(a: Output, x: Output, name: String = "Igamma"): Output =
    Op.Builder(opType = "Igamma", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)

  def zeta(x: Output, q: Output, name: String = "Zeta"): Output =
    Op.Builder(opType = "Zeta", name = name)
        .addInput(x)
        .addInput(q)
        .build().outputs(0)

  def polygamma(a: Output, x: Output, name: String = "Polygamma"): Output =
    Op.Builder(opType = "Polygamma", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)

  //endregion Binary Ops

  def betainc(a: Output, b: Output, x: Output, name: String = "Betainc"): Output =
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
  def logicalNot(x: Output, name: String = "LogicalNot"): Output = {
    Op.Builder(opType = "LogicalNot", name = name)
        .addInput(x)
        .build().outputs(0)
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
  def logicalAnd(x: Output, y: Output, name: String = "LogicalAnd"): Output = {
    Op.Builder(opType = "LogicalAnd", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
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
  def logicalOr(x: Output, y: Output, name: String = "LogicalOr"): Output = {
    Op.Builder(opType = "LogicalOr", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
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
  def logicalXor(x: Output, y: Output, name: String = "LogicalXor"): Output = {
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
  def equal(x: Output, y: Output, name: String = "Equal"): Output = {
    Op.Builder(opType = "Equal", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
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
  def notEqual(x: Output, y: Output, name: String = "NotEqual"): Output = {
    Op.Builder(opType = "NotEqual", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the truth value of `abs(x - y) < tolerance`  element-wise.
    *
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  tolerance Comparison tolerance value.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def approximatelyEqual(
      x: Output, y: Output, tolerance: Float = 0.00001f, name: String = "ApproximatelyEqual"): Output = {
    Op.Builder(opType = "ApproximateEqual", name = name)
        .addInput(x)
        .addInput(y)
        .setAttribute("tolerance", tolerance)
        .build().outputs(0)
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
  def less(x: Output, y: Output, name: String = "Less"): Output = {
    Op.Builder(opType = "Less", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
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
  def lessEqual(x: Output, y: Output, name: String = "LessEqual"): Output = {
    Op.Builder(opType = "LessEqual", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
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
  def greater(x: Output, y: Output, name: String = "Greater"): Output = {
    Op.Builder(opType = "Greater", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
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
  def greaterEqual(x: Output, y: Output, name: String = "GreaterEqual"): Output = {
    Op.Builder(opType = "GreaterEqual", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  //endregion Comparison Ops

  //region Reduction Ops

  private[this] def reductionAxes(tensor: Output, axes: Output): Output = {
    if (axes != null)
      axes
    else
      Basic.constant(Tensor.fromSeq(0 until tensor.shape.rank: _*)(INT32.supportedType))
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
    */
  def sum(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Sum"): Output = {
    Op.Builder(opType = "Sum", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
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
  def mean(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Mean"): Output = {
    Op.Builder(opType = "Mean", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
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
  def product(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Product"): Output = {
    Op.Builder(opType = "Prod", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
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
    */
  def min(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Min"): Output = {
    Op.Builder(opType = "Min", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
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
    */
  def max(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Max"): Output = {
    Op.Builder(opType = "Max", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
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
    */
  def all(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "All"): Output = {
    Op.Builder(opType = "All", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
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
    */
  def any(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Any"): Output = {
    Op.Builder(opType = "Any", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
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
  def logSumExp(
      input: Output, axes: Array[Int] = null, keepDims: Boolean = false, name: String = "LogSumExp"): Output = {
    // TODO: !!! Can we support a dynamic version for the axes argument?
    Op.createWith(nameScope = name) {
      val maxValue = Basic.stopGradient(max(input, axes, keepDims = true))
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
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the dimensions to reduce. If `null`, then all dimensions are reduced.
    * @param  keepDims If `true`, retain the reduced dimensions.
    * @param  name     Name for the created op.
    * @return Created op output with `INT64` data type.
    */
  def countNonZero(
      input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "CountNonZero"): Output = {
    Op.createWith(nameScope = name) {
      sum(cast(notEqual(input, Basic.constant(0)), INT64), axes, keepDims)
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
  def segmentSum(data: Output, segmentIndices: Output, name: String = "SegmentSum"): Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentSum", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
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
  def segmentMean(data: Output, segmentIndices: Output, name: String = "SegmentMean"): Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentMean", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
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
  def segmentProd(data: Output, segmentIndices: Output, name: String = "SegmentProd"): Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentProd", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
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
  def segmentMin(data: Output, segmentIndices: Output, name: String = "SegmentMin"): Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentMin", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
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
  def segmentMax(data: Output, segmentIndices: Output, name: String = "SegmentMax"): Output = {
    if (!data.dataType.isNumeric)
      throw InvalidDataTypeException(s"'data' data type, '${data.dataType}', is not a numeric data type, as required.")
    if (segmentIndices.dataType != INT32 && segmentIndices.dataType != INT64)
      throw InvalidDataTypeException(
        s"'segmentIndices' data type, '${segmentIndices.dataType}', is not 'INT32' or 'INT64', as required.")
    Op.Builder(opType = "SegmentMax", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
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
  def unsortedSegmentSum(
      data: Output, segmentIndices: Output, segmentsNumber: Output, name: String = "UnsortedSegmentSum"): Output = {
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
        .build().outputs(0)
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
  def unsortedSegmentProd(
      data: Output, segmentIndices: Output, segmentsNumber: Output, name: String = "UnsortedSegmentMax"): Output = {
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
        .build().outputs(0)
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

    private[this] def diagGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(diagPart(outputGradients.head))
    }

    private[this] def diagPartGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(diag(outputGradients.head))
    }

    private[this] def matrixDiagGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(matrixDiagPart(outputGradients.head))
    }

    private[this] def matrixSetDiagGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val gradient = outputGradients.head
      val inputShape = op.inputs(0).shape.mergeWith(gradient.shape)
      val batchShape = inputShape(0 :: -2).mergeWith(op.inputs(1).shape(0 :: -1))
      val matrixShape = inputShape(-2 ::)
      val diagShape = {
        if (batchShape.isFullyDefined && matrixShape.isFullyDefined) {
          Basic.constant(Tensor((batchShape.asArray :+ matrixShape.asArray.min).map(Tensor(_)): _*))
        } else {
          Op.colocateWith(Set(gradient.op)) {
            val gradShape = Basic.shape(gradient)
            val gradRank = Basic.rank(gradient)
            val batchShape = Basic.slice(gradShape, 0, gradRank - 2)
            val matrixShape = Basic.slice(gradShape, gradRank - 2, 2)
            val minDim = min(matrixShape)
            Basic.concatenate(Seq(batchShape, minDim), 0)
          }
        }
      }
      val gradInput = matrixSetDiag(gradient, Basic.fill(diagShape, Tensor(gradient.dataType, 0)))
      val gradDiag = matrixDiagPart(gradient)
      Seq(gradInput, gradDiag)
    }

    private[this] def matrixDiagPartGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val matrixShape = op.inputs(0).shape(-2 ::)
      if (matrixShape.isFullyDefined && matrixShape(0) == matrixShape(1))
        Seq(matrixDiag(outputGradients.head))
      else
        Seq(matrixSetDiag(Basic.zerosLike(op.inputs(0)), outputGradients.head))
    }

    private[this] def matrixBandPartGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(matrixBandPart(outputGradients.head, op.inputs(1), op.inputs(2)), null, null)
    }

    private[this] def castGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val supportedDataTypes = Seq(FLOAT32, FLOAT64) // TODO: [TYPES] Float16 and complex.
      val sourceDataType = op.inputs(0).dataType
      val destinationDataType = outputGradients.head.dataType
      if (supportedDataTypes.contains(sourceDataType) && supportedDataTypes.contains(destinationDataType))
        Seq(cast(outputGradients.head, sourceDataType))
      else
        Seq(null)
    }

    private[this] def matMulGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      matMulGradientCommon(op, outputGradients, "transpose_a", "transpose_b", isBatch = false)
    }

    private[this] def batchMatMulGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      matMulGradientCommon(op, outputGradients, "adj_x", "adj_y", isBatch = true)
    }

    private[this] def matMulGradientCommon(
        op: Op, outputGradients: Seq[OutputLike], transposeAAttribute: String, transposeBAttribute: String,
        isBatch: Boolean): Seq[OutputLike] = {
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
        x0: Output, x1: Output, y0: Output, y1: Output, transposeX0: Boolean, transposeX1: Boolean,
        transposeY0: Boolean, transposeY1: Boolean, isBatch: Boolean): Seq[OutputLike] = {
      if (!isBatch) {
        val gradientX = matMul(x0, x1, transposeA = transposeX0, transposeB = transposeX1, name = "MatMul_1")
        val gradientY = matMul(y0, y1, transposeA = transposeY0, transposeB = transposeY1, name = "MatMul_2")
        Seq[OutputLike](gradientX, gradientY)
      } else {
        val gradientX = batchMatMul(x0, x1, adjointX = transposeX0, adjointY = transposeX1, name = "MatMul_1")
        val gradientY = batchMatMul(y0, y1, adjointX = transposeY0, adjointY = transposeY1, name = "MatMul_2")
        Seq[OutputLike](gradientX, gradientY)
      }
    }

    private[this] def squareGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      var x = op.inputs(0)
      val outputGradient = outputGradients.head
      // Using control dependencies to prevent 2*x from being computed too early.
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        x = conjugate(x)
        // TODO: !!! Automatic casting for mathematic operations? At least print nicely formatted exception?
        Seq(outputGradient * (Basic.constant(2, x.dataType) * x))
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
    private[this] def broadcastGradientArguments(
        shape0: Output, shape1: Output, name: String = "BroadcastGradientArguments"): (Output, Output) = {
      val outputs = Op.Builder(opType = "BroadcastGradientArgs", name = name)
          .addInput(shape0)
          .addInput(shape1)
          .build().outputs
      (outputs(0), outputs(1))
    }

    private[this] def subtractGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
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

    private[this] def reduceSumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
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
