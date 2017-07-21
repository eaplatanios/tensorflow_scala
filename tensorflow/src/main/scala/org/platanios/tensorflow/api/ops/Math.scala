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
import org.platanios.tensorflow.api.types.{DataType, FLOAT32, FLOAT64, COMPLEX64, COMPLEX128, INT32, INT64}

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
    * @param  start Rank 0 (i.e., scalar) tensor that contains the starting value of the number sequence.
    * @param  limit Rank 0 (i.e., scalar) tensor that contains the ending value (exclusive) of the number sequence.
    * @param  delta Rank 0 (i.e., scalar) tensor that contains the difference between consecutive numbers in the
    *               sequence.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def range(
      start: Output, limit: Output, delta: Output = Basic.constant(1), dataType: DataType = null,
      name: String = "Range"): Output = {
    require(start.rank == 0, s"'start' (rank = ${start.rank}) must have rank 0 (i.e., must be a scalar tensor).")
    require(limit.rank == 0, s"'limit' (rank = ${limit.rank}) must have rank 0 (i.e., must be a scalar tensor).")
    require(delta.rank == 0, s"'delta' (rank = ${delta.rank}) must have rank 0 (i.e., must be a scalar tensor).")
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

  /** Creates an op that generates values in an interval.
    *
    * The op generates a sequence of `numberOfValues` evenly-spaced values beginning at `start`. If
    * `numberOfValues > 1`, the values in the sequence increase by `(stop - start) / (numberOfValues - 1)`, so that the
    * last value is exactly equal to `stop`.
    *
    * For example:
    * {{{
    *   linspace(10.0, 12.0, 3) ==> [10.0  11.0  12.0]
    * }}}
    *
    * @param  start          Rank 0 (i.e., scalar) tensor that contains the starting value of the number sequence.
    * @param  stop           Rank 0 (i.e., scalar) tensor that contains the ending value (inclusive) of the number
    *                        sequence.
    * @param  numberOfValues Rank 0 (i.e., scalar) tensor that contains the number of values in the number sequence.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def linspace(start: Output, stop: Output, numberOfValues: Output, name: String = "LinSpace"): Output = {
    require(start.rank == 0, s"'start' (rank = ${start.rank}) must have rank 0 (i.e., must be a scalar tensor).")
    require(stop.rank == 0, s"'stop' (rank = ${stop.rank}) must have rank 0 (i.e., must be a scalar tensor).")
    require(numberOfValues.rank == 0,
            s"'numberOfValues' (rank = ${numberOfValues.rank}) must have rank 0 (i.e., must be a scalar tensor).")
    require(start.dataType == FLOAT32 || start.dataType == FLOAT64,
            s"Unsupported 'start' data type '${start.dataType}'. Must be 'FLOAT32' or 'FLOAT64'.")
    require(stop.dataType == FLOAT32 || stop.dataType == FLOAT64,
            s"Unsupported 'stop' data type '${stop.dataType}'. Must be 'FLOAT32' or 'FLOAT64'.")
    require(numberOfValues.dataType == INT32 || numberOfValues.dataType == INT64,
            s"Unsupported 'numberOfValues' data type '${numberOfValues.dataType}'. Must be 'INT32' or 'INT64'.")
    Op.Builder(opType = "LinSpace", name = name)
        .addInput(start)
        .addInput(stop)
        .addInput(numberOfValues)
        .build().outputs(0)
  }

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
  def cast[T: OutputOps](x: T, dataType: DataType, name: String = "Cast"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Cast", name = name)
            .addInput(o)
            .setAttribute("DstT", dataType)
            .build().outputs(0))
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

  /** Creates an op that adds all input tensors element-wise.
    *
    * @param  inputs Input tensors (must all have the same shape and size).
    * @param  name   Created op name.
    * @return Created op output.
    */
  def addN(inputs: Array[Output], name: String = "AddN"): Output = {
    require(inputs.length > 0, "'inputs' must contain at least one tensor.")
    if (inputs.length == 1) {
      Basic.identity(inputs(0), name)
    } else {
      Op.Builder(opType = "AddN", name = name)
          .addInputList(inputs)
          .build().outputs(0)
    }
  }

  // TODO: [OPS] accumulateN

  //region Unary Ops

  /** Creates an op that computes the absolute value of a tensor.
    *
    * Given a tensor `x`, the op returns a tensor containing the absolute value of each element in `x`. For example, if
    * `x` is an input element and `y` is an output element, the op computes `y = |x|`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def abs[T: OutputOps](x: T, name: String = "Abs"): T = {
    // TODO: [COMPLEX]
    implicitly[OutputOps[T]]
        .unaryOp(x, o =>
          Op.Builder(opType = "Abs", name = name)
              .addInput(o)
              .build().outputs(0))
  }

  def complexAbs[T : OutputOps](x: T, name: String = "ComplexAbs"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "ComplexAbs", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the numerical negative value of a tensor element-wise.
    *
    * I.e., `y = -x`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def negate[T: OutputOps](x: T, name: String = "Negate"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Neg", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the reciprocal value of a tensor element-wise.
    *
    * I.e., `y = 1 / x`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def reciprocal[T: OutputOps](x: T, name: String = "Reciprocal"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Reciprocal", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the square of a tensor element-wise.
    *
    * I.e., `y = x * x = x^2`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def square[T: OutputOps](x: T, name: String = "Square"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Square", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the square root of a tensor element-wise.
    *
    * I.e., `y = \sqrt{x} = x^{1/2}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sqrt[T: OutputOps](x: T, name: String = "Sqrt"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Sqrt", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the reciprocal of the square root of a tensor element-wise.
    *
    * I.e., `y = 1 / \sqrt{x} = 1 / x^{1/2}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def rsqrt[T: OutputOps](x: T, name: String = "Rsqrt"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Rsqrt", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the exponential of a tensor element-wise.
    *
    * I.e., `y = \exp{x} = e^x`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def exp[T: OutputOps](x: T, name: String = "Exp"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Exp", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the exponential of a tensor minus `1` element-wise.
    *
    * I.e., `y = \exp{x} - 1`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def expm1[T: OutputOps](x: T, name: String = "Expm1"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Expm1", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the logarithm of a tensor element-wise.
    *
    * I.e., `y = \log{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def log[T: OutputOps](x: T, name: String = "Log"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Log", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the logarithm of a tensor plus `1` element-wise.
    *
    * I.e., `y = \log{1 + x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def log1p[T: OutputOps](x: T, name: String = "Log1p"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Log1p", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the sine of a tensor element-wise.
    *
    * I.e., `y = \sin{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sin[T: OutputOps](x: T, name: String = "Sin"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Sin", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the cosine of a tensor element-wise.
    *
    * I.e., `y = \cos{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def cos[T: OutputOps](x: T, name: String = "Cos"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Cos", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the tangent of a tensor element-wise.
    *
    * I.e., `y = \tan{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def tan[T: OutputOps](x: T, name: String = "Tan"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Tan", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the inverse sine of a tensor element-wise.
    *
    * I.e., `y = \asin{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def asin[T: OutputOps](x: T, name: String = "Asin"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Asin", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the inverse cosine of a tensor element-wise.
    *
    * I.e., `y = \acos{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def acos[T: OutputOps](x: T, name: String = "Acos"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Acos", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the inverse tangent of a tensor element-wise.
    *
    * I.e., `y = \atan{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def atan[T: OutputOps](x: T, name: String = "Atan"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Atan", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the hyperbolic sine of a tensor element-wise.
    *
    * I.e., `y = \sinh{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sinh[T: OutputOps](x: T, name: String = "Sinh"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Sinh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the hyperbolic cosine of a tensor element-wise.
    *
    * I.e., `y = \cosh{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def cosh[T: OutputOps](x: T, name: String = "Cosh"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Cosh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the hyperbolic tangent of a tensor element-wise.
    *
    * I.e., `y = \tanh{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def tanh[T: OutputOps](x: T, name: String = "Tanh"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Tanh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the inverse hyperbolic sine of a tensor element-wise.
    *
    * I.e., `y = \asinh{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def asinh[T: OutputOps](x: T, name: String = "ASinh"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Asinh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the inverse hyperbolic cosine of a tensor element-wise.
    *
    * I.e., `y = \acosh{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def acosh[T: OutputOps](x: T, name: String = "ACosh"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Acosh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the inverse hyperbolic tangent of a tensor element-wise.
    *
    * I.e., `y = \atanh{x}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def atanh[T: OutputOps](x: T, name: String = "ATanh"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Atanh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the logarithm of the absolute value of the Gamma function applied element-wise on a
    * tensor.
    *
    * I.e., `y = \log{|\Gamma{x}|}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logGamma[T: OutputOps](x: T, name: String = "Lgamma"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Lgamma", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the derivative of the logarithm of the absolute value of the Gamma function applied
    * element-wise on a tensor (i.e., the digamma or Psi function).
    *
    * I.e., `y = \partial\log{|\Gamma{x}|}`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def digamma[T: OutputOps](x: T, name: String = "Digamma"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Digamma", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the Gaussian error function element-wise on a tensor.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def erf[T: OutputOps](x: T, name: String = "Erf"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Erf", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the complementary Gaussian error function element-wise on a tensor.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def erfc[T: OutputOps](x: T, name: String = "Erfc"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Erfc", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the sigmoid function element-wise on a tensor.
    *
    * I.e., `y = 1 / (1 + \exp{-x})`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sigmoid[T: OutputOps](x: T, name: String = "Sigmoid"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Sigmoid", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  // TODO: [OPS] logSigmoid

  /** Creates an op that returns an element-wise indication of the sign of a tensor.
    *
    * I.e., `y = sign(x) = -1` if `x < 0`; `0` if `x == 0`; `1` if `x > 0`. For complex numbers,
    * `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sign[T: OutputOps](x: T, name: String = "Sign"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Sign", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the round value of a tensor element-wise.
    *
    * Rounds half to even. Also known as bankers rounding. If you want to round according to the current system rounding
    * mode use the [[roundInt]] op instead.
    *
    * For example:
    * {{{
    *   // 'a' is [0.9, 2.5, 2.3, 1.5, -4.5]
    *   round(a) ==> [1.0, 2.0, 2.0, 2.0, -4.0]
    * }}}
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `COMPLEX64`, or
    *              `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def round[T: OutputOps](x: T, name: String = "Round"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Round", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the round value of a tensor element-wise.
    *
    * If the result is midway between two representable values, the even representable is chosen.
    *
    * For example:
    * {{{
    *   roundInt(-1.5) ==> -2.0
    *   roundInt(0.5000001) ==> 1.0
    *   roundInt([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
    * }}}
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def roundInt[T: OutputOps](x: T, name: String = "RoundInt"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Rint", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the largest integer not greater than the current value of a tensor, element-wise.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def floor[T: OutputOps](x: T, name: String = "Floor"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Floor", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that computes the smallest integer not greater than the current value of a tensor, element-wise.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def ceil[T: OutputOps](x: T, name: String = "Ceil"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "Ceil", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that returns a boolean tensor indicating which elements of a tensor are NaN-valued.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def isNaN[T: OutputOps](x: T, name: String = "IsNaN"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "IsNan", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that returns a boolean tensor indicating which elements of a tensor are Inf-valued.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def isInf[T: OutputOps](x: T, name: String = "IsInf"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "IsInf", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** Creates an op that returns a boolean tensor indicating which elements of a tensor are finite-valued.
    *
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def isFinite[T: OutputOps](x: T, name: String = "IsFinite"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(x, o => Op.Builder(opType = "IsFinite", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  //endregion Unary Ops

  //region Binary Ops

  /** Creates an op that adds two tensors element-wise.
    *
    * I.e., `z = x + y`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, `COMPLEX128`, or `STRING`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, `COMPLEX128`, or `STRING`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def add(x: Output, y: Output, name: String = "Add"): Output = {
    Op.Builder(opType = "Add", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that subtracts two tensors element-wise.
    *
    * I.e., `z = x - y`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def subtract(x: Output, y: Output, name: String = "Sub"): Output = {
    Op.Builder(opType = "Sub", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that multiplies two tensors element-wise.
    *
    * I.e., `z = x * y`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def multiply(x: Output, y: Output, name: String = "Mul"): Output = {
    Op.Builder(opType = "Mul", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that divides two tensors element-wise.
    *
    * I.e., `z = x / y`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def divide(x: Output, y: Output, name: String = "Div"): Output = {
    Op.Builder(opType = "Div", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that floor-divides two tensors element-wise.
    *
    * I.e., `z = x // y`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  @deprecated("Use `truncateDivide` instead.")
  def floorDivide(x: Output, y: Output, name: String = "FloorDiv"): Output = {
    Op.Builder(opType = "FloorDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that truncate-divides two integer tensors element-wise.
    *
    * Truncation designates that negative numbers will round fractional quantities toward zero. I.e. `-7 / 5 = 1`. This
    * matches C semantics but it is different than Python semantics. See `floorDivide` for a division function that
    * matches Python semantics.
    *
    * I.e., `z = x / y`, for `x` and `y` being integer tensors.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def truncateDivide(x: Output, y: Output, name: String = "TruncateDiv"): Output = {
    Op.Builder(opType = "TruncateDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that divides two real tensors element-wise.
    *
    * If `x` and `y` are real-valued tensors, the op will return the floating-point division.
    *
    * I.e., `z = x / y`, for `x` and `y` being real tensors.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def realDivide(x: Output, y: Output, name: String = "RealDiv"): Output = {
    Op.Builder(opType = "RealDiv", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the squared difference between two tensors element-wise.
    *
    * I.e., `z = (x - y) * (x - y)`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def squaredDifference(x: Output, y: Output, name: String = "SquaredDifference"): Output = {
    Op.Builder(opType = "SquaredDifference", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the remainder of the division between two tensors element-wise.
    *
    * The op emulates C semantics in that the result is consistent with a truncating divide.
    * E.g., `truncate(x / y) * y + truncateMod(x, y) = x`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def mod(x: Output, y: Output, name: String = "Mod"): Output = {
    Op.Builder(opType = "Mod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the remainder of the division between two tensors element-wise.
    *
    * When `x < 0` xor `y < 0` is true, the op follows Python semantics in that the result here is consistent with a
    * flooring divide. E.g., `floor(x / y) * y + mod(x, y) = x`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def floorMod(x: Output, y: Output, name: String = "FloorMod"): Output = {
    Op.Builder(opType = "FloorMod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the remainder of the division between two tensors element-wise.
    *
    * The op emulates C semantics in that the result here is consistent with a truncating divide.
    * E.g., `truncate(x / y) * y + truncateMod(x, y) = x`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def truncateMod(x: Output, y: Output, name: String = "TruncateMod"): Output = {
    Op.Builder(opType = "TruncateMod", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the power of one tensor raised to another, element-wise.
    *
    * Given a tensor `x` and a tensor `y`, the op computes `x^y` for the corresponding elements in `x` and `y`.
    *
    * For example:
    * {{{
    *   // Tensor 'x' is [[2, 2], [3, 3]]
    *   // Tensor 'y' is [[8, 16], [2, 3]]
    *   pow(x, y) ==> [[256, 65536], [9, 27]]
    * }}}
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def pow(x: Output, y: Output, name: String = "Pow"): Output = {
    Op.Builder(opType = "Pow", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that computes the upper regularized incomplete Gamma function `Q(a, x)`.
    *
    * The upper regularized incomplete Gamma function is defined as:
    *
    * `Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)`, where:
    *
    * `Gamma(a, x) = \int_{x}^{\infty} t^{a-1} exp(-t) dt`
    *
    * is the upper incomplete Gama function.
    *
    * Note that, above, `P(a, x)` (`Igamma`) is the lower regularized complete Gamma function.
    *
    * @param  a    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def igammac(a: Output, x: Output, name: String = "Igammac"): Output = {
    Op.Builder(opType = "Igammac", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)
  }

  /** Creates an op that computes the lower regularized incomplete Gamma function `Q(a, x)`.
    *
    * The lower regularized incomplete Gamma function is defined as:
    *
    * `P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)`, where:
    *
    * `Gamma(a, x) = \int_{0}^{x} t^{a-1} exp(-t) dt`
    *
    * is the lower incomplete Gamma function.
    *
    * Note that, above, `Q(a, x)` (`Igammac`) is the upper regularized complete Gamma function.
    *
    * @param  a    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def igamma(a: Output, x: Output, name: String = "Igamma"): Output = {
    Op.Builder(opType = "Igamma", name = name)
        .addInput(a)
        .addInput(x)
        .build().outputs(0)
  }

  /** Creates an op that computes the Hurwitz zeta function `\zeta(x, q)`.
    *
    * The Hurwitz zeta function is defined as:
    *
    * `\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}`.
    *
    * @param  x    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  q    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def zeta(x: Output, q: Output, name: String = "Zeta"): Output = {
    Op.Builder(opType = "Zeta", name = name)
        .addInput(x)
        .addInput(q)
        .build().outputs(0)
  }

  /** Creates an op that computes the polygamma function `\psi^{(n)}(x)`.
    *
    * The polygamma function is defined as:
    *
    * `\psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)`, where `\psi(x)` is the digamma function.
    *
    * @param  n    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def polygamma(n: Output, x: Output, name: String = "Polygamma"): Output = {
    Op.Builder(opType = "Polygamma", name = name)
        .addInput(n)
        .addInput(x)
        .build().outputs(0)
  }

  /** Creates an op that computes the inverse tangent of `y / x` element-wise, respecting signs of the arguments.
    *
    * The op computes the angle `\theta \in [-\pi, \pi]` such that `x = r \cos(\theta)` and `y = r \sin(\theta)`, where
    * `r = \sqrt(x^2 + y^2)`.
    *
    * @param  y    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def atan2(y: Output, x: Output, name: String = "ATan2"): Output = {
    Op.Builder(opType = "Atan2", name = name)
        .addInput(y)
        .addInput(x)
        .build().outputs(0)
  }

  /** Creates an op that returns the element-wise maximum between two tensors.
    *
    * I.e., `z = x > y ? x : y`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              or `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def maximum(x: Output, y: Output, name: String = "Maximum"): Output = {
    Op.Builder(opType = "Maximum", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  /** Creates an op that returns the element-wise minimum between two tensors.
    *
    * I.e., `z = x < y ? x : y`.
    *
    * NOTE: This op supports broadcasting. More information about broadcasting can be found
    * [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    *
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              or `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def minimum(x: Output, y: Output, name: String = "Minimum"): Output = {
    Op.Builder(opType = "Minimum", name = name)
        .addInput(x)
        .addInput(y)
        .build().outputs(0)
  }

  //endregion Binary Ops

  /** Creates an op that computes the regularized incomplete beta integral `I_x(a, b)`.
    *
    * The regularized incomplete beta integral is defined as:
    *
    * `I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}`, where:
    *
    * `B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt`
    *
    * is the incomplete beta function and `B(a, b)` is the *complete* beta function.
    *
    * @param  a    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  b    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x    Third input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def incompleteBeta(a: Output, b: Output, x: Output, name: String = "IncompleteBeta"): Output = {
    Op.Builder(opType = "Betainc", name = name)
        .addInput(a)
        .addInput(b)
        .addInput(x)
        .build().outputs(0)
  }

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
  def logicalXOr(x: Output, y: Output, name: String = "LogicalXOr"): Output = {
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

  private[this] def reductionAxes[T <: OutputLike](tensor: T, axes: Output): Output = {
    if (axes != null) {
      axes
    } else {
      tensor match { // Fast path: Avoid creating range and rank ops if the rank is known statically.
        case t: Output if t.rank > -1 =>
          Basic.constant(Tensor.fromSeq(0 until t.rank: _*)(INT32.supportedType))
        case o: SparseOutput if o.denseShape.shape.isFullyDefined =>
          Basic.constant(Tensor.fromSeq(0 until o.denseShape.shape(0): _*)(INT32.supportedType))
        case _ => // Otherwise, we rely on range and rank to do the right thing at run-time.
          range(0, Basic.rank(tensor))
      }
    }
  }

  /** Creates an op that computes the sum of elements across axes of a tensor.
    *
    * Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced by
    * 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    * If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
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
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
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

  /** Creates an op that computes the mean of elements across axes of a tensor.
    *
    * Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced by
    * 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    * If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
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
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
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

  /** Creates an op that computes the product of elements across axes of a tensor.
    *
    * Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced by
    * 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    * If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    * For example:
    * {{{
    *   // 'x' is [[1, 1, 1]], [1, 1, 1]]
    *   prod(x) == 1
    *   prod(x, 0) == [1, 1, 1]
    *   prod(x, 1) == [1, 1]
    *   prod(x, 1, keepDims = true) == [[1], [1]]
    *   prod(x, [0, 1]) == 1
    * }}}
    *
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def prod(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Prod"): Output = {
    Op.Builder(opType = "Prod", name = name)
        .addInput(input)
        .addInput(reductionAxes(input, axes))
        .setAttribute("keep_dims", keepDims)
        .build().outputs(0)
  }

  /** Creates an op that computes the minimum of elements across axes of a tensor.
    *
    * Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced by
    * 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    * If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
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
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
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

  /** Creates an op that computes the maximum of elements across axes of a tensor.
    *
    * Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced by
    * 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    * If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
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
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
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

  /** Creates an op that computes the logical AND of elements across axes of a tensor.
    *
    * Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced by
    * 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    * If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
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
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
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

  /** Creates an op that computes the logical OR of elements across axes of a tensor.
    *
    * Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced by
    * 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    * If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
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
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
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

  /** Creates an op that computes the log-sum-exp of elements across axes of a tensor.
    *
    * Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced by
    * 1 for each entry in `axes`. If `keepDims` is `true`, the reduced dimensions are retained with size 1.
    *
    * If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
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
    * @param  axes     Integer array containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def logSumExp(
      input: Output, axes: Array[Int] = null, keepDims: Boolean = false, name: String = "LogSumExp"): Output = {
    Op.createWith(nameScope = name) {
      val maxValue = Basic.stopGradient(max(input, axes, keepDims = true))
      val result = log(sum(exp(input - maxValue), axes, keepDims = true)) + maxValue
      if (keepDims)
        result
      else
        Basic.squeeze(result, axes)
    }
  }

  /** Creates an op that computes the number of non-zero elements across axes of a tensor.
    *
    * Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced by
    * 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    * If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
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
    * @param  axes     Integer array containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
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

  /** Creates an op that returns the indices with the largest value across axes of a tensor.
    *
    * Note that in case of ties the identity of the return value is not guaranteed.
    *
    * @param  input          Input tensor.
    * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor. Must be `INT32` or `INT64`.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def argmax(input: Output, axes: Output = 0, outputDataType: DataType = INT64, name: String = "ArgMax"): Output = {
    if (axes.dataType != INT32 && axes.dataType != INT64)
      throw new IllegalArgumentException(s"'axes.dataType' (${axes.dataType}) must be INT32 or INT64.")
    if (outputDataType != INT32 && outputDataType != INT64)
      throw new IllegalArgumentException(s"'outputDataType' ($outputDataType) must be INT32 or INT64.")
    Op.Builder(opType = "ArgMax", name = name)
        .addInput(input)
        .addInput(axes)
        .setAttribute("output_type", outputDataType)
        .build().outputs(0)
  }

  /** Creates an op that returns the indices with the smallest value across axes of a tensor.
    *
    * Note that in case of ties the identity of the return value is not guaranteed.
    *
    * @param  input          Input tensor.
    * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor. Must be `INT32` or `INT64`.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def argmin(input: Output, axes: Output = 0, outputDataType: DataType = INT64, name: String = "ArgMin"): Output = {
    if (axes.dataType != INT32 && axes.dataType != INT64)
      throw new IllegalArgumentException(s"'axes.dataType' (${axes.dataType}) must be INT32 or INT64.")
    if (outputDataType != INT32 && outputDataType != INT64)
      throw new IllegalArgumentException(s"'outputDataType' ($outputDataType) must be INT32 or INT64.")
    Op.Builder(opType = "ArgMin", name = name)
        .addInput(input)
        .addInput(axes)
        .setAttribute("output_type", outputDataType)
        .build().outputs(0)
  }

  /** Creates an op that counts the number of occurrences of each value in an integer tensor.
    *
    * If `minLength` and `maxLength` are not provided, the op returns a vector with length `max(input) + 1`, if `input`
    * is non-empty, and length `0` otherwise.
    *
    * If `weights` is not `null`, then index `i` of the output stores the sum of the value in `weights` at each index
    * where the corresponding value in `input` is equal to `i`.
    *
    * @param  input     `INT32` tensor containing non-negative values.
    * @param  weights   If not `null`, this tensor must have the same shape as `input`. For each value in `input`, the
    *                   corresponding bin count will be incremented by the corresponding weight instead of `1`.
    * @param  minLength If not `null`, this ensures the output has length at least `minLength`, padding with zeros at
    *                   the end, if necessary.
    * @param  maxLength If not `null`, this skips values in `input` that are equal or greater than `maxLength`, ensuring
    *                   that the output has length at most `maxLength`.
    * @param  dataType  If `weights` is `null`, this determines the data type used for the output tensor (i.e., the
    *                   tensor containing the bin counts).
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def binCount(
      input: Output, weights: Output = null, minLength: Output = null, maxLength: Output = null,
      dataType: DataType = INT32, name: String = "BinCount"): Output = {
    require(input.dataType == INT32, s"'input' (dataType = ${input.dataType}) must have INT32 data type.")
    val inputNonEmpty = greater(prod(Basic.shape(input)), 0)
    var outputSize = cast(inputNonEmpty, INT32) * (max(input) + 1)
    if (minLength != null)
      outputSize = maximum(minLength, outputSize)
    if (maxLength != null)
      outputSize = minimum(maxLength, outputSize)
    val effectiveWeights = {
      if (weights != null) {
        weights
      } else {
        Basic.zeros(Shape(), dataType)
      }
    }
    Op.Builder(opType = "Bincount", name = name)
        .addInput(input)
        .addInput(outputSize)
        .addInput(effectiveWeights)
        .build().outputs(0)
  }

  /** Creates an op that computes the cumulative sum of the tensor along an axis.
    *
    * By default, the op performs an inclusive cumulative sum, which means that the first element of the input is
    * identical to the first element of the output:
    * {{{
    *   cumsum([a, b, c]) ==> [a, a + b, a + b + c]
    * }}}
    *
    * By setting the `exclusive` argument to `true`, an exclusive cumulative sum is performed instead:
    * {{{
    *   cumsum([a, b, c], exclusive = true) ==> [0, a, a + b]
    * }}}
    *
    * By setting the `reverse` argument to `true`, the cumulative sum is performed in the opposite direction:
    * {{{
    *   cumsum([a, b, c], reverse = true) ==> [a + b + c, b + c, c]
    * }}}
    *
    * This is more efficient than using separate [[Basic.reverse]] ops.
    *
    * The `reverse` and `exclusive` arguments can also be combined:
    * {{{
    *   cumsum([a, b, c], exclusive = true, reverse = true) ==> [b + c, c, 0]
    * }}}
    *
    * @param  input     Input tensor.
    * @param  axis      `INT32` tensor containing the axis along which to perform the cumulative sum.
    * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
    * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def cumsum(
      input: Output, axis: Output = 0, exclusive: Boolean = false, reverse: Boolean = false,
      name: String = "CumSum"): Output = {
    require(axis.dataType == INT32, s"'axis' (dataType = ${axis.dataType}) must have 'INT32' data type.")
    Op.Builder(opType = "Cumsum", name = name)
        .addInput(input)
        .addInput(axis)
        .setAttribute("exclusive", exclusive)
        .setAttribute("reverse", reverse)
        .build().outputs(0)
  }

  /** Creates an op that computes the cumulative product of the tensor along an axis.
    *
    * By default, the op performs an inclusive cumulative product, which means that the first element of the input is
    * identical to the first element of the output:
    * {{{
    *   cumprod([a, b, c]) ==> [a, a * b, a * b * c]
    * }}}
    *
    * By setting the `exclusive` argument to `true`, an exclusive cumulative product is performed instead:
    * {{{
    *   cumprod([a, b, c], exclusive = true) ==> [0, a, a * b]
    * }}}
    *
    * By setting the `reverse` argument to `true`, the cumulative product is performed in the opposite direction:
    * {{{
    *   cumprod([a, b, c], reverse = true) ==> [a * b * c, b * c, c]
    * }}}
    *
    * This is more efficient than using separate [[Basic.reverse]] ops.
    *
    * The `reverse` and `exclusive` arguments can also be combined:
    * {{{
    *   cumprod([a, b, c], exclusive = true, reverse = true) ==> [b * c, c, 0]
    * }}}
    *
    * @param  input     Input tensor.
    * @param  axis      `INT32` tensor containing the axis along which to perform the cumulative product.
    * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative product.
    * @param  reverse   Boolean value indicating whether to perform a reverse cumulative product.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def cumprod(
      input: Output, axis: Output = 0, exclusive: Boolean = false, reverse: Boolean = false,
      name: String = "CumProd"): Output = {
    require(axis.dataType == INT32, s"'axis' (dataType = ${axis.dataType}) must have 'INT32' data type.")
    Op.Builder(opType = "Cumprod", name = name)
        .addInput(input)
        .addInput(axis)
        .setAttribute("exclusive", exclusive)
        .setAttribute("reverse", reverse)
        .build().outputs(0)
  }

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

  //region Matrix Ops

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

  /** Creates an op that computes the trace of a tensor.
    *
    * The trace of a tensor is defined as the sum along the main diagonal of each inner-most matrix in it. If the tensor
    * is of rank `k` with shape `[I, J, K, ..., L, M, N]`, then output is a tensor of rank `k - 2` with dimensions
    * `[I, J, K, ..., L]` where: `output[i, j, k, ..., l] = trace(x[i, j, i, ..., l, :, :])`.
    *
    * For example:
    * {{{
    *   // Tensor 'x' is [[1, 2], [3, 4]]
    *   trace(x) ==> 5
    *
    *   // Tensor 'x' is [[1, 2, 3],
    *                     [4, 5, 6],
    *                     [7, 8, 9]]
    *   trace(x) ==> 15
    *
    *   // Tensor 'x' is [[[ 1,  2,  3],
    *                      [ 4,  5,  6],
    *                      [ 7,  8,  9]],
    *                     [[-1, -2, -3],
    *                      [-4, -5, -6],
    *                      [-7, -8, -9]]]
    *   trace(x) ==> [15, -15]
    * }}}
    *
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def trace(input: Output, name: String = "Trace"): Output = {
    Op.createWithNameScope(name) {
      sum(matrixDiagPart(input), axes = -1)
    }
  }

  // TODO: !!! matMul documentation plus sparse matMul.

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

  /** Creates an op that computes the pairwise cross product between two tensors.
    *
    * `a` and `b` must have the same shape; they can either be simple 3-element vectors, or have any shape where the
    * innermost dimension size is 3. In the latter case, each pair of corresponding 3-element vectors is
    * cross-multiplied independently.
    *
    * @param  a    First input tensor.
    * @param  b    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def cross(a: Output, b: Output, name: String = "Cross"): Output = {
    require(a.dataType == b.dataType,
            s"'a' (dataType = ${a.dataType}) and 'b' (dataType = ${b.dataType}) must have the same data type.")
    require(a.shape == b.shape, s"'a' (shape = ${a.shape}) and 'b' (shape = ${b.shape}) must have the same shape.")
    require(a.shape(-1) == 3, s"The inner-most dimension size of the shape of 'a' and 'b' (${a.shape}) must be 3.")
    Op.Builder(opType = "Cross", name = name)
        .addInput(a)
        .addInput(b)
        .build().outputs(0)
  }

  // TODO: [OPS] tensorDot

  //endregion Matrix Ops

  //region Complex Ops

  /** Creates an op that converts two real tensors to a complex tensor.
    *
    * Given a tensor `real` representing the real part of a complex number, and a tensor `imag` representing the
    * imaginary part of a complex number, the op returns complex numbers element-wise of the form `a + bj`, where *a*
    * represents the `real` part and *b* represents the `imag` part. The input tensors `real` and `imag` must have the
    * same shape and data type.
    *
    * For example:
    * {{{
    *   // Tensor 'real' is [2.25, 3.25]
    *   // Tensor 'imag' is [4.75, 5.75]
    *   complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
    * }}}
    *
    * @param  real Tensor containing the real component. Must have `FLOAT32` or `FLOAT64` data type.
    * @param  imag Tensor containing the imaginary component. Must have `FLOAT32` or `FLOAT64` data type.
    * @param  name Name for the created op.
    * @return Created op output with data type being either `COMPLEX64` or `COMPLEX128`.
    * @throws IllegalArgumentException If 'real' and 'imag' have different shapes or invalid data types.
    */
  @throws[IllegalArgumentException]
  def complex(real: Output, imag: Output, name: String = "Complex"): Output = {
    if (real.shape != imag.shape)
      throw new IllegalArgumentException(
        s"'real' (shape = ${real.shape}) and 'imag' (shape = ${imag.shape}) must have the same shape.")
    val outputDataType = (real.dataType, imag.dataType) match {
      case (FLOAT32, FLOAT32) => COMPLEX64
      case (FLOAT64, FLOAT64) => COMPLEX128
      case _ => throw new IllegalArgumentException(
        s"'real' (dataType = ${real.dataType}) and 'imag' (dataType = ${imag.dataType}) must both have the same data " +
            s"type, which must be either 'FLOAT32' or 'FLOAT64'.")
    }
    Op.Builder(opType = "Complex", name = name)
        .addInput(real)
        .addInput(imag)
        .setAttribute("Tout", outputDataType)
        .build().outputs(0)
  }

  /** Creates an op that returns the real part of a complex number.
    *
    * Given a tensor `input` of potentially complex numbers, the op returns a tensor of type `FLOAT32` or `FLOAT64` that
    * is the real part of each element in `input`. If `input` contains complex numbers of the form `a + bj`, *a* is the
    * real part returned by the op and *b* is the imaginary part.
    *
    * For example:
    * {{{
    *   // Tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *   real(input) ==> [-2.25, 3.25]
    * }}}
    *
    * Note that, if `input` is already real-valued, then it is returned unchanged.
    *
    * @param  input Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def real(input: Output, name: String = "Real"): Output = {
    if (!input.dataType.isComplex) {
      input
    } else {
      Op.Builder(opType = "Real", name = name)
          .addInput(input)
          .build().outputs(0)
    }
  }

  /** Creates an op that returns the real part of a complex number.
    *
    * Given a tensor `input` of complex numbers, the op returns a tensor of type `FLOAT32` or `FLOAT64` that is the
    * imaginary part of each element in `input`. If `input` contains complex numbers of the form `a + bj`, *a* is the
    * real part and *b* is the imaginary part returned by the op.
    *
    * For example:
    * {{{
    *   // Tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *   real(input) ==> [4.75, 5.75]
    * }}}
    *
    * @param  input Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def imag(input: Output, name: String = "Imag"): Output = {
    if (!input.dataType.isComplex) {
      input
    } else {
      Op.Builder(opType = "Imag", name = name)
          .addInput(input)
          .setAttribute("Tout", input.dataType.real)
          .build().outputs(0)
    }
  }

  /** Creates an op that returns the element-wise complex conjugate of a tensor.
    *
    * Given a numeric tensor `input`, the op returns a tensor with numbers that are the complex conjugate of each
    * element in `input`. If the numbers in `input` are of the form `a + bj`, where *a* is the real part and *b* is the
    * imaginary part, then the complex conjugate returned by this operation is of the form `a - bj`.
    *
    * For example:
    * {{{
    *   // Tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *   conjugate(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
    * }}}
    *
    * If `input` is real-valued, then it is returned unchanged.
    *
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If the provided tensor is not numeric.
    */
  @throws[IllegalArgumentException]
  def conjugate[T <: OutputLike : OutputOps](input: T, name: String = "Conjugate"): T = {
    implicitly[OutputOps[T]]
        .unaryOp(input, o => {
          if (input.dataType.isComplex) {
            Op.Builder(opType = "Conj", name = name)
                .addInput(o)
                .build().outputs(0)
          } else if (input.dataType.isNumeric) {
            o
          } else {
            throw new IllegalArgumentException("'conjugate' can only take numeric tensors as input.")
          }
        })
  }

  //endregion Complex Ops

  //region Quantization Ops

  // TODO: [OPS] quantization

  //endregion Quantization Ops
}

object Math extends Math {
  private[api] object Gradients {
    GradientsRegistry.register("Select", selectGradient)
    GradientsRegistry.register("Cast", castGradient)
    GradientsRegistry.register("AddN", addNGradient)
    // TODO: Abs gradient.
    GradientsRegistry.register("Neg", negateGradient)
    GradientsRegistry.register("Reciprocal", reciprocalGradient)
    GradientsRegistry.register("Square", squareGradient)
    GradientsRegistry.register("Sqrt", sqrtGradient)
    GradientsRegistry.register("Rsqrt", rsqrtGradient)
    GradientsRegistry.register("Tanh", tanhGradient)
    GradientsRegistry.register("Sigmoid", sigmoidGradient)
    GradientsRegistry.register("Sub", subtractGradient)
    GradientsRegistry.register("Sum", sumGradient)
    GradientsRegistry.register("Diag", diagGradient)
    GradientsRegistry.register("DiagPart", diagPartGradient)
    GradientsRegistry.register("MatrixDiag", matrixDiagGradient)
    GradientsRegistry.register("MatrixSetDiag", matrixSetDiagGradient)
    GradientsRegistry.register("MatrixDiagPart", matrixDiagPartGradient)
    GradientsRegistry.register("MatrixBandPart", matrixBandPartGradient)
    GradientsRegistry.register("MatMul", matMulGradient)
    GradientsRegistry.register("BatchMatMul", batchMatMulGradient)
    GradientsRegistry.register("Conj", conjugateGradient)

    GradientsRegistry.registerNonDifferentiable("Range")
    GradientsRegistry.registerNonDifferentiable("LinSpace")
    // Logical operations have no gradients.
    GradientsRegistry.registerNonDifferentiable("LogicalNot")
    GradientsRegistry.registerNonDifferentiable("LogicalAnd")
    GradientsRegistry.registerNonDifferentiable("LogicalOr")
    GradientsRegistry.registerNonDifferentiable("Equal")
    GradientsRegistry.registerNonDifferentiable("NotEqual")
    GradientsRegistry.registerNonDifferentiable("ApproximateEqual")
    GradientsRegistry.registerNonDifferentiable("Less")
    GradientsRegistry.registerNonDifferentiable("LessEqual")
    GradientsRegistry.registerNonDifferentiable("Greater")
    GradientsRegistry.registerNonDifferentiable("GreaterEqual")

    private[this] def selectGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val grad = outputGradients.head
      val c = op.inputs(0)
      val x = op.inputs(1)
      val zeros = Basic.zerosLike(x)
      Seq[OutputLike](null, select(c, grad, zeros), select(c, zeros, grad))
    }

    private[this] def unaryGradientOp(
        y: Output, outputGradients: Seq[OutputLike], opType: String, name: String): Seq[OutputLike] = {
      val outputGradient = outputGradients.head
      val gradient = outputGradient match {
        case g: Output =>
          Op.Builder(opType = opType, name = name)
              .addInput(y)
              .addInput(g)
              .build().outputs(0)
        case g: OutputIndexedSlices =>
          val values = Op.Builder(opType = opType, name = name)
              .addInput(y)
              .addInput(g)
              .build().outputs(0)
          OutputIndexedSlices(indices = g.indices, values = values, denseShape = g.denseShape)
        case g: SparseOutput =>
          val values = Op.Builder(opType = opType, name = name)
              .addInput(y)
              .addInput(g)
              .build().outputs(0)
          SparseOutput(indices = g.indices, values = values, denseShape = g.denseShape)
      }
      Seq(gradient)
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

    private[this] def addNGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq.fill(op.numInputs)(outputGradients.head)
    }

    // TODO: !!! absGradient.

    private[this] def negateGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(negate(outputGradients.head))
    }

    private[this] def reciprocalGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      unaryGradientOp(op.outputs(0), outputGradients, opType = "ReciprocalGrad", name = "ReciprocalGradient")
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

    private[this] def sqrtGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      unaryGradientOp(op.outputs(0), outputGradients, opType = "SqrtGrad", name = "SqrtGradient")
    }

    private[this] def rsqrtGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      unaryGradientOp(op.outputs(0), outputGradients, opType = "RsqrtGrad", name = "RSqrtGradient")
    }

    private[this] def tanhGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      var y = op.outputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        y = conjugate(y)
        unaryGradientOp(y, outputGradients, opType = "TanhGrad", name = "TanhGradient")
      }
    }

    private[this] def sigmoidGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      var y = op.outputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        y = conjugate(y)
        unaryGradientOp(y, outputGradients, opType = "SigmoidGrad", name = "SigmoidGradient")
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

    private[this] def sumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
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

    private[this] def conjugateGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(conjugate(outputGradients.head))
    }
  }
}
