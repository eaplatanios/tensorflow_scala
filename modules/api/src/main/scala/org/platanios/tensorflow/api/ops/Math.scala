/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.tensors
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._

import scala.language.postfixOps

/** Contains functions for constructing general math-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Math {
  /** $OpDocMathSelect
    *
    * @group MathOps
    * @param  condition Boolean condition tensor.
    * @param  x         Tensor which may have the same shape as `condition`. If `condition` has rank `1`, then `t` may
    *                   have a higher rank, but its first dimension must match the size of `condition`.
    * @param  y         Tensor with the same data type and shape as `t`.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def select(condition: Output, x: Output, y: Output, name: String = "Select"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Select", name = name)
        .addInput(condition)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathRange
    *
    * @group MathOps
    * @param  start Rank 0 (i.e., scalar) tensor that contains the starting value of the number sequence.
    * @param  limit Rank 0 (i.e., scalar) tensor that contains the ending value (exclusive) of the number sequence.
    * @param  delta Rank 0 (i.e., scalar) tensor that contains the difference between consecutive numbers in the
    *               sequence.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def range(
      start: Output,
      limit: Output,
      delta: Output = Basic.constant(1),
      dataType: DataType = null,
      name: String = "Range"
  ): Output = {
    var castedStart: Output = start
    var castedLimit: Output = limit
    var castedDelta: Output = delta
    Op.createWith(nameScope = name) {
      val inferredDataType = {
        if (dataType != null)
          dataType
        else
          DataType.mostPrecise(start.dataType, limit.dataType, delta.dataType)
      }
      if (start.dataType != inferredDataType)
        castedStart = Cast.cast(start, inferredDataType)
      if (limit.dataType != inferredDataType)
        castedLimit = Cast.cast(limit, inferredDataType)
      if (delta.dataType != inferredDataType)
        castedDelta = Cast.cast(delta, inferredDataType)
    }
    Op.Builder(opType = "Range", name = name)
        .addInput(castedStart)
        .addInput(castedLimit)
        .addInput(castedDelta)
        .build().outputs(0)
  }

  /** $OpDocMathLinspace
    *
    * @group MathOps
    * @param  start          Rank 0 (i.e., scalar) tensor that contains the starting value of the number sequence.
    * @param  stop           Rank 0 (i.e., scalar) tensor that contains the ending value (inclusive) of the number
    *                        sequence.
    * @param  numberOfValues Rank 0 (i.e., scalar) tensor that contains the number of values in the number sequence.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def linspace(start: Output, stop: Output, numberOfValues: Output, name: String = "LinSpace"): Output = {
    Op.Builder(opType = "LinSpace", name = name)
        .addInput(start)
        .addInput(stop)
        .addInput(numberOfValues)
        .build().outputs(0)
  }

  /** $OpDocMathAddN
    *
    * @group MathOps
    * @param  inputs Input tensors.
    * @param  name   Created op name.
    * @return Created op output.
    */
  def addN(inputs: Seq[Output], name: String = "AddN"): Output = {
    if (inputs.length == 1)
      Basic.identity(inputs(0), name)
    else
      Op.Builder(opType = "AddN", name = name)
          .addInputList(castArgs(inputs))
          .build().outputs(0)
  }

  /** $OpDocMathAccumulateN
    *
    * @param  inputs Input tensors.
    * @param  shape  Shape of the elements of `inputs` (in case it's not known statically and needs to be retained).
    * @param  name   Created op name.
    * @return Created op output.
    * @throws InvalidArgumentException If any of the inputs has a different data type and/or shape than the rest.
    */
  @throws[InvalidArgumentException]
  def accumulateN(
      inputs: Seq[Output],
      shape: Shape = null,
      name: String = "AccumulateN"
  ): Output = {
    val dataType = inputs.head.dataType
    if (inputs.exists(_.dataType != dataType))
      throw InvalidArgumentException("All input tensors must have the same data type.")
    val inferredShape = if (shape == null) Shape.unknown() else shape
    if (inputs.exists(!_.shape.isCompatibleWith(inferredShape)))
      throw InvalidArgumentException("All input tensors must have the same shape.")
    if (inputs.length == 1 && name == null) {
      inputs.head
    } else if (inputs.length == 1) {
      Basic.identity(inputs.head, name = name)
    } else {
      Op.Builder(opType = "AccumulateNV2", name = name)
          .addInputList(inputs)
          .setAttribute("shape", shape)
          .build().outputs(0)
    }
  }

  //region Unary Ops

  /** $OpDocMathAbs
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def abs[T <: OutputLike : OutputOps](x: T, name: String = "Abs"): T = {
    if (x.dataType.isComplex) {
      implicitly[OutputOps[T]]
          .applyUnary(x, o => Op.Builder(opType = "ComplexAbs", name = name)
              .addInput(o)
              .setAttribute("Tout", x.dataType.real)
              .build().outputs(0))
    } else {
      implicitly[OutputOps[T]]
          .applyUnary(x, o =>
            Op.Builder(opType = "Abs", name = name)
                .addInput(o)
                .build().outputs(0))
    }
  }

  /** $OpDocMathNegate
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def negate[T: OutputOps](x: T, name: String = "Negate"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Neg", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathReciprocal
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def reciprocal[T: OutputOps](x: T, name: String = "Reciprocal"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Reciprocal", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathSquare
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def square[T: OutputOps](x: T, name: String = "Square"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Square", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathSqrt
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sqrt[T: OutputOps](x: T, name: String = "Sqrt"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Sqrt", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathRsqrt
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def rsqrt[T: OutputOps](x: T, name: String = "Rsqrt"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Rsqrt", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathExp
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def exp[T: OutputOps](x: T, name: String = "Exp"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Exp", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathExpm1
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def expm1[T: OutputOps](x: T, name: String = "Expm1"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Expm1", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathLog
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def log[T: OutputOps](x: T, name: String = "Log"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Log", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathLog1p
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def log1p[T: OutputOps](x: T, name: String = "Log1p"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Log1p", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathSin
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sin[T: OutputOps](x: T, name: String = "Sin"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Sin", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathCos
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def cos[T: OutputOps](x: T, name: String = "Cos"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Cos", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathTan
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def tan[T: OutputOps](x: T, name: String = "Tan"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Tan", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathAsin
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def asin[T: OutputOps](x: T, name: String = "Asin"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Asin", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathAcos
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def acos[T: OutputOps](x: T, name: String = "Acos"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Acos", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathAtan
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def atan[T: OutputOps](x: T, name: String = "Atan"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Atan", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathSinh
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sinh[T: OutputOps](x: T, name: String = "Sinh"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Sinh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathCosh
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def cosh[T: OutputOps](x: T, name: String = "Cosh"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Cosh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathTanh
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def tanh[T: OutputOps](x: T, name: String = "Tanh"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Tanh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathAsinh
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def asinh[T: OutputOps](x: T, name: String = "ASinh"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Asinh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathAcosh
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def acosh[T: OutputOps](x: T, name: String = "ACosh"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Acosh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathAtanh
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def atanh[T: OutputOps](x: T, name: String = "ATanh"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Atanh", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathLogGamma
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logGamma[T: OutputOps](x: T, name: String = "Lgamma"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Lgamma", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathDigamma
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def digamma[T: OutputOps](x: T, name: String = "Digamma"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Digamma", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathErf
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def erf[T: OutputOps](x: T, name: String = "Erf"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Erf", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathErfc
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def erfc[T: OutputOps](x: T, name: String = "Erfc"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Erfc", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathSigmoid
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sigmoid[T: OutputOps](x: T, name: String = "Sigmoid"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Sigmoid", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathLogSigmoid
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logSigmoid[T: OutputOps](x: T, name: String = "LogSigmoid"): T = {
    Op.createWithNameScope(name) {
      negate(NN.softplus(negate(x)))
    }
  }

  /** $OpDocMathSign
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *              `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def sign[T: OutputOps](x: T, name: String = "Sign"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Sign", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathRound
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `COMPLEX64`, or
    *              `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def round[T: OutputOps](x: T, name: String = "Round"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Round", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathRoundInt
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def roundInt[T: OutputOps](x: T, name: String = "RoundInt"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Rint", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathFloor
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def floor[T: OutputOps](x: T, name: String = "Floor"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Floor", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathCeil
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def ceil[T: OutputOps](x: T, name: String = "Ceil"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "Ceil", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathIsNaN
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def isNaN[T: OutputOps](x: T, name: String = "IsNaN"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "IsNan", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathIsInf
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def isInf[T: OutputOps](x: T, name: String = "IsInf"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "IsInf", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  /** $OpDocMathIsFinite
    *
    * @group MathOps
    * @param  x    Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def isFinite[T: OutputOps](x: T, name: String = "IsFinite"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(x, o => Op.Builder(opType = "IsFinite", name = name)
            .addInput(o)
            .build().outputs(0))
  }

  //endregion Unary Ops

  //region Binary Ops

  /** $OpDocMathAdd
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, `COMPLEX128`, or `STRING`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, `COMPLEX128`, or `STRING`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def add(x: Output, y: Output, name: String = "Add"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Add", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathSubtract
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def subtract(x: Output, y: Output, name: String = "Sub"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Sub", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathMultiply
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def multiply(x: Output, y: Output, name: String = "Mul"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Mul", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathDivide
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def divide(x: Output, y: Output, name: String = "Div"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Div", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathFloorDivide
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  @deprecated("Use `truncateDivide` instead.", "0.1")
  def floorDivide(x: Output, y: Output, name: String = "FloorDiv"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "FloorDiv", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathTruncateDivide
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def truncateDivide(x: Output, y: Output, name: String = "TruncateDiv"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "TruncateDiv", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathRealDivide
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *              `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def realDivide(x: Output, y: Output, name: String = "RealDiv"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "RealDiv", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathSquaredDifference
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def squaredDifference(x: Output, y: Output, name: String = "SquaredDifference"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "SquaredDifference", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathMod
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def mod(x: Output, y: Output, name: String = "Mod"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Mod", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathFloorMod
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def floorMod(x: Output, y: Output, name: String = "FloorMod"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "FloorMod", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathTruncateMod
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def truncateMod(x: Output, y: Output, name: String = "TruncateMod"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "TruncateMod", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathPow
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def pow(x: Output, y: Output, name: String = "Pow"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Pow", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathIgammac
    *
    * @group MathOps
    * @param  a    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def igammac(a: Output, x: Output, name: String = "Igammac"): Output = {
    val (cA, cX) = castArgs(a, x)
    Op.Builder(opType = "Igammac", name = name)
        .addInput(cA)
        .addInput(cX)
        .build().outputs(0)
  }

  /** $OpDocMathIgamma
    *
    * @group MathOps
    * @param  a    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def igamma(a: Output, x: Output, name: String = "Igamma"): Output = {
    val (cA, cX) = castArgs(a, x)
    Op.Builder(opType = "Igamma", name = name)
        .addInput(cA)
        .addInput(cX)
        .build().outputs(0)
  }

  /** $OpDocMathZeta
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  q    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def zeta(x: Output, q: Output, name: String = "Zeta"): Output = {
    val (cX, cQ) = castArgs(x, q)
    Op.Builder(opType = "Zeta", name = name)
        .addInput(cX)
        .addInput(cQ)
        .build().outputs(0)
  }

  /** $OpDocMathPolygamma
    *
    * @group MathOps
    * @param  n    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def polygamma(n: Output, x: Output, name: String = "Polygamma"): Output = {
    val (cN, cX) = castArgs(n, x)
    Op.Builder(opType = "Polygamma", name = name)
        .addInput(cN)
        .addInput(cX)
        .build().outputs(0)
  }

  /** $OpDocMathAtan2
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  y    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def atan2(x: Output, y: Output, name: String = "ATan2"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Atan2", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathMinimum
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              or `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def minimum(x: Output, y: Output, name: String = "Minimum"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Minimum", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathMaximum
    *
    * @group MathOps
    * @param  x    First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, or
    *              `INT64`.
    * @param  y    Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *              or `INT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def maximum(x: Output, y: Output, name: String = "Maximum"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Maximum", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  //endregion Binary Ops

  /** $OpDocMathIncompleteBeta
    *
    * @group MathOps
    * @param  a    First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  b    Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x    Third input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def incompleteBeta(a: Output, b: Output, x: Output, name: String = "IncompleteBeta"): Output = {
    val (cA, cB, cX) = castArgs(a, b, x)
    Op.Builder(opType = "Betainc", name = name)
        .addInput(cA)
        .addInput(cB)
        .addInput(cX)
        .build().outputs(0)
  }

  //region Logical Ops

  /** $OpDocMathLogicalNot
    *
    * @group MathOps
    * @param  x    Input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def logicalNot(x: Output, name: String = "LogicalNot"): Output = {
    Op.Builder(opType = "LogicalNot", name = name)
        .addInput(x)
        .build().outputs(0)
  }

  /** $OpDocMathLogicalAnd
    *
    * @group MathOps
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

  /** $OpDocMathLogicalOr
    *
    * @group MathOps
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

  /** $OpDocMathLogicalXOr
    *
    * @group MathOps
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

  /** $OpDocMathEqual
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def equal(x: Output, y: Output, name: String = "Equal"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Equal", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathNotEqual
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def notEqual(x: Output, y: Output, name: String = "NotEqual"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "NotEqual", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** $OpDocMathApproximatelyEqual
    *
    * @group MathOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  tolerance Comparison tolerance value.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def approximatelyEqual(
      x: Output, y: Output, tolerance: Float = 0.00001f, name: String = "ApproximatelyEqual"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "ApproximateEqual", name = name)
        .addInput(cX)
        .addInput(cY)
        .setAttribute("tolerance", tolerance)
        .build().outputs(0)
  }

  /** OpDocMathLess
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def less(x: Output, y: Output, name: String = "Less"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Less", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** OpDocMathLessEqual
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def lessEqual(x: Output, y: Output, name: String = "LessEqual"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "LessEqual", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** OpDocMathGreater
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def greater(x: Output, y: Output, name: String = "Greater"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "Greater", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  /** OpDocMathGreaterEqual
    *
    * @group MathOps
    * @param  x    First input tensor.
    * @param  y    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def greaterEqual(x: Output, y: Output, name: String = "GreaterEqual"): Output = {
    val (cX, cY) = castArgs(x, y)
    Op.Builder(opType = "GreaterEqual", name = name)
        .addInput(cX)
        .addInput(cY)
        .build().outputs(0)
  }

  //endregion Comparison Ops

  //region Reduction Ops

  private[this] def reductionAxes[T <: OutputLike](tensor: T, axes: Output): Output = {
    if (axes != null) {
      axes
    } else {
      tensor match { // Fast path: Avoid creating range and rank ops if the rank is known statically.
        case o: Output if o.rank == 0 =>
          Basic.constant(Tensor.zeros(INT32, Shape(0)))
        case o: Output if o.rank > -1 =>
          Basic.constant(0 until o.rank)
        case o: OutputIndexedSlices if o.denseShape.shape.isFullyDefined =>
          Basic.constant(0 until o.denseShape.shape(0))
        case o: SparseOutput if o.denseShape.shape.isFullyDefined =>
          Basic.constant(0 until o.denseShape.shape(0))
        case _ => // Otherwise, we rely on range and rank to do the right thing at run-time.
          range(0, Basic.rank(tensor))
      }
    }
  }

  /** $OpDocMathSum
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def sum(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Sum"): Output = {
    if (input.rank == 0)
      input
    else
      Op.Builder(opType = "Sum", name = name)
          .addInput(input)
          .addInput(reductionAxes(input, axes))
          .setAttribute("keep_dims", keepDims)
          .build().outputs(0)
  }

  /** $OpDocMathMean
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def mean(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Mean"): Output = {
    if (input.rank == 0)
      input
    else
      Op.Builder(opType = "Mean", name = name)
          .addInput(input)
          .addInput(reductionAxes(input, axes))
          .setAttribute("keep_dims", keepDims)
          .build().outputs(0)
  }

  /** $OpDocMathProd
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def prod(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Prod"): Output = {
    if (input.rank == 0)
      input
    else
      Op.Builder(opType = "Prod", name = name)
          .addInput(input)
          .addInput(reductionAxes(input, axes))
          .setAttribute("keep_dims", keepDims)
          .build().outputs(0)
  }

  /** $OpDocMathMin
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def min(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Min"): Output = {
    if (input.rank == 0)
      input
    else
      Op.Builder(opType = "Min", name = name)
          .addInput(input)
          .addInput(reductionAxes(input, axes))
          .setAttribute("keep_dims", keepDims)
          .build().outputs(0)
  }

  /** $OpDocMathMax
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def max(input: Output, axes: Output = null, keepDims: Boolean = false, name: String = "Max"): Output = {
    if (input.rank == 0)
      input
    else
      Op.Builder(opType = "Max", name = name)
          .addInput(input)
          .addInput(reductionAxes(input, axes))
          .setAttribute("keep_dims", keepDims)
          .build().outputs(0)
  }

  /** $OpDocMathAll
    *
    * @group MathOps
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

  /** $OpDocMathAny
    *
    * @group MathOps
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

  /** $OpDocMathLogSumExp
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def logSumExp(
      input: Output,
      axes: Output = null,
      keepDims: Boolean = false,
      name: String = "LogSumExp"
  ): Output = {
    if (input.rank == 0)
      input
    else
      Op.createWith(nameScope = name) {
        val maxValue = Basic.stopGradient(max(input, axes, keepDims = true))
        var result = log(sum(exp(input - maxValue), axes, keepDims = keepDims))
        if (!keepDims)
          result += Basic.reshape(maxValue, Basic.shape(result))
        else
          result += maxValue
        result
      }
  }

  /** $OpDocMathCountNonZero
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Created op output with `INT64` data type.
    */
  def countNonZero(
      input: Output,
      axes: Output = null,
      keepDims: Boolean = false,
      name: String = "CountNonZero"
  ): Output = {
    Op.createWith(nameScope = name) {
      sum(Cast.cast(notEqual(input, Basic.zeros(input.dataType, Shape())), INT64), axes, keepDims)
    }
  }

  /** $OpDocMathCountNonZero
    *
    * @group MathOps
    * @param  input    Input tensor for which to count the number of non-zero entries.
    * @param  name     Name for the created op.
    * @return Created op output with `INT64` data type.
    */
  def countNonZeroSparse[T <: OutputLike](input: T, name: String = "CountNonZero"): Output = {
    Op.createWith(nameScope = name) {
      val zero = Basic.zeros(input.dataType, Shape())
      input match {
        case o: Output => sum(Cast.cast(notEqual(o, zero), INT64))
        case o: OutputIndexedSlices => sum(Cast.cast(notEqual(o.values, zero), INT64))
        case o: SparseOutput => sum(Cast.cast(notEqual(o.values, zero), INT64))
      }
    }
  }

  //endregion Reduction Ops

  /** $OpDocMathArgmax
    *
    * @group MathOps
    * @param  input          Input tensor.
    * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor. Must be `INT32` or `INT64`.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def argmax(input: Output, axes: Output = 0, outputDataType: DataType = INT64, name: String = "ArgMax"): Output = {
    Op.Builder(opType = "ArgMax", name = name)
        .addInput(input)
        .addInput(axes)
        .setAttribute("output_type", outputDataType)
        .build().outputs(0)
  }

  /** $OpDocMathArgmin
    *
    * @group MathOps
    * @param  input          Input tensor.
    * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor. Must be [[INT32]] or [[INT64]].
    * @param  name           Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `axes` data type or `outputDataType` is not [[INT32]] or [[INT64]].
    */
  @throws[IllegalArgumentException]
  def argmin(input: Output, axes: Output = 0, outputDataType: DataType = INT64, name: String = "ArgMin"): Output = {
    Op.Builder(opType = "ArgMin", name = name)
        .addInput(input)
        .addInput(axes)
        .setAttribute("output_type", outputDataType)
        .build().outputs(0)
  }

  /** $OpDocMathBinCount
    *
    * @group MathOps
    * @param  input     [[INT32]] tensor containing non-negative values.
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
    val inputNonEmpty = greater(prod(Basic.shape(input)), 0)
    var outputSize = Cast.cast(inputNonEmpty, INT32) * (max(input) + 1)
    if (minLength != null)
      outputSize = maximum(minLength, outputSize)
    if (maxLength != null)
      outputSize = minimum(maxLength, outputSize)
    val effectiveWeights = {
      if (weights != null) {
        weights
      } else {
        Basic.zeros(dataType, Shape.scalar())
      }
    }
    Op.Builder(opType = "Bincount", name = name)
        .addInput(input)
        .addInput(outputSize)
        .addInput(effectiveWeights)
        .build().outputs(0)
  }

  /** $OpDocMathCumsum
    *
    * @group MathOps
    * @param  input     Input tensor.
    * @param  axis      [[INT32]] tensor containing the axis along which to perform the cumulative sum.
    * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
    * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
    * @param  name      Name for the created op.
    * @return Created op output.
    */
  def cumsum(
      input: Output, axis: Output = 0, exclusive: Boolean = false, reverse: Boolean = false,
      name: String = "CumSum"): Output = {
    Op.Builder(opType = "Cumsum", name = name)
        .addInput(input)
        .addInput(axis)
        .setAttribute("exclusive", exclusive)
        .setAttribute("reverse", reverse)
        .build().outputs(0)
  }

  /** $OpDocMathCumprod
    *
    * @group MathOps
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
    Op.Builder(opType = "Cumprod", name = name)
        .addInput(input)
        .addInput(axis)
        .setAttribute("exclusive", exclusive)
        .setAttribute("reverse", reverse)
        .build().outputs(0)
  }

  //region Segment Ops

  /** $OpDocMathSegmentSum
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentSum(data: Output, segmentIndices: Output, name: String = "SegmentSum"): Output = {
    Op.Builder(opType = "SegmentSum", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** $OpDocMathSegmentMean
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentMean(data: Output, segmentIndices: Output, name: String = "SegmentMean"): Output = {
    Op.Builder(opType = "SegmentMean", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** $OpDocMathSegmentProd
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentProd(data: Output, segmentIndices: Output, name: String = "SegmentProd"): Output = {
    Op.Builder(opType = "SegmentProd", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** $OpDocMathSegmentMin
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentMin(data: Output, segmentIndices: Output, name: String = "SegmentMin"): Output = {
    Op.Builder(opType = "SegmentMin", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** $OpDocMathSegmentMax
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def segmentMax(data: Output, segmentIndices: Output, name: String = "SegmentMax"): Output = {
    Op.Builder(opType = "SegmentMax", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .build().outputs(0)
  }

  /** $OpDocMathUnsortedSegmentSum
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
    * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentSum(
      data: Output,
      segmentIndices: Output,
      segmentsNumber: Output,
      name: String = "UnsortedSegmentSum"
  ): Output = {
    Op.Builder(opType = "UnsortedSegmentSum", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .addInput(segmentsNumber)
        .build().outputs(0)
  }

  /** Helper function for `unsortedSegmentMean` and `unsortedSegmentSqrtN` that computes the number of segment entries
    * with zero entries set to `1`, in order to allow for division by `N`.
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
    * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
    * @return Created op output.
    */
  protected def unsortedSegmentN(
      data: Output,
      segmentIndices: Output,
      segmentsNumber: Output,
      name: String = "UnsortedSegmentN"
  ): Output = Op.createWithNameScope(name) {
    // `binCount` does not support negative indices and so we use `unsortedSegmentSum`.
    val ones = Basic.ones(data.dataType, Basic.shape(segmentIndices))
    val N = unsortedSegmentSum(ones, segmentIndices, segmentsNumber)
    val outputRank = Basic.rank(data) - Basic.rank(segmentIndices)
    val outputRankTiled = Basic.tile(Basic.ones(segmentsNumber.dataType, Shape(1)), outputRank.expandDims(0))
    val broadcastShape = Basic.concatenate(Seq(segmentsNumber.expandDims(0), outputRankTiled))
    maximum(1, Basic.reshape(N, broadcastShape))
  }

  /** $OpDocMathUnsortedSegmentMean
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
    * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentMean(
      data: Output,
      segmentIndices: Output,
      segmentsNumber: Output,
      name: String = "UnsortedSegmentMean"
  ): Output = Op.createWithNameScope(name) {
    val N = unsortedSegmentN(data, segmentIndices, segmentsNumber, name = "N")
    unsortedSegmentSum(data, segmentIndices, segmentsNumber, name = "Sum") / N
  }

  /** $OpDocMathUnsortedSegmentProd
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
    * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentProd(
      data: Output,
      segmentIndices: Output,
      segmentsNumber: Output,
      name: String = "UnsortedSegmentProd"
  ): Output = {
    Op.Builder(opType = "UnsortedSegmentProd", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .addInput(segmentsNumber)
        .build().outputs(0)
  }

  /** $OpDocMathUnsortedSegmentMin
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
    * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentMin(
      data: Output, segmentIndices: Output, segmentsNumber: Output, name: String = "UnsortedSegmentMin"): Output = {
    Op.Builder(opType = "UnsortedSegmentMin", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .addInput(segmentsNumber)
        .build().outputs(0)
  }

  /** $OpDocMathUnsortedSegmentMax
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
    * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentMax(
      data: Output, segmentIndices: Output, segmentsNumber: Output, name: String = "UnsortedSegmentMax"): Output = {
    Op.Builder(opType = "UnsortedSegmentMax", name = name)
        .addInput(data)
        .addInput(segmentIndices)
        .addInput(segmentsNumber)
        .build().outputs(0)
  }

  /** $OpDocMathUnsortedSegmentSqrtN
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
    * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def unsortedSegmentSqrtN(
      data: Output,
      segmentIndices: Output,
      segmentsNumber: Output,
      name: String = "UnsortedSegmentSqrtN"
  ): Output = Op.createWithNameScope(name) {
    val N = unsortedSegmentN(data, segmentIndices, segmentsNumber, name = "N")
    unsortedSegmentSum(data, segmentIndices, segmentsNumber, name = "Sum") / sqrt(N)
  }

  /** $OpDocMathSparseSegmentSum
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  numSegments    Optional `INT32` scalar indicating the size of the output tensor.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def sparseSegmentSum(
      data: Output, indices: Output, segmentIndices: Output, numSegments: Output = null,
      name: String = "SparseSegmentSum"): Output = {
    if (numSegments == null) {
      Op.Builder(opType = "SparseSegmentSum", name = name)
          .addInput(data)
          .addInput(indices)
          .addInput(segmentIndices)
          .build().outputs(0)
    } else {
      Op.Builder(opType = "SparseSegmentSumWithNumSegments", name = name)
          .addInput(data)
          .addInput(indices)
          .addInput(segmentIndices)
          .addInput(numSegments)
          .build().outputs(0)
    }
  }

  /** $OpDocMathSparseSegmentMean
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  numSegments    Optional `INT32` scalar indicating the size of the output tensor.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def sparseSegmentMean(
      data: Output, indices: Output, segmentIndices: Output, numSegments: Output = null,
      name: String = "SparseSegmentMean"): Output = {
    if (numSegments == null) {
      Op.Builder(opType = "SparseSegmentMean", name = name)
          .addInput(data)
          .addInput(indices)
          .addInput(segmentIndices)
          .build().outputs(0)
    } else {
      Op.Builder(opType = "SparseSegmentMeanWithNumSegments", name = name)
          .addInput(data)
          .addInput(indices)
          .addInput(segmentIndices)
          .addInput(numSegments)
          .build().outputs(0)
    }
  }

  /** $OpDocMathSparseSegmentSumSqrtN
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  numSegments    Optional `INT32` scalar indicating the size of the output tensor.
    * @param  name           Name for the created op.
    * @return Created op output.
    */
  def sparseSegmentSumSqrtN(
      data: Output, indices: Output, segmentIndices: Output, numSegments: Output = null,
      name: String = "SparseSegmentSumSqrtN"): Output = {
    if (numSegments == null) {
      Op.Builder(opType = "SparseSegmentSqrtN", name = name)
          .addInput(data)
          .addInput(indices)
          .addInput(segmentIndices)
          .build().outputs(0)
    } else {
      Op.Builder(opType = "SparseSegmentSqrtNWithNumSegments", name = name)
          .addInput(data)
          .addInput(indices)
          .addInput(segmentIndices)
          .addInput(numSegments)
          .build().outputs(0)
    }
  }

  //endregion Segment Ops

  //region Matrix Ops

  /** $OpDocMathDiag
    *
    * @group MathOps
    * @param  diagonal Diagonal values, represented as a rank-`K` tensor, where `K` can be at most `3`.
    * @param  name     Name for the created op.
    * @return Created op output.
    */
  def diag(diagonal: Output, name: String = "Diag"): Output = {
    Op.Builder(opType = "Diag", name = name)
        .addInput(diagonal)
        .build().outputs(0)
  }

  /** $OpDocMathDiagPart
    *
    * @group MathOps
    * @param  input Rank-`K` input tensor, where `K` is either `2`, `4`, or `6`.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def diagPart(input: Output, name: String = "DiagPart"): Output = {
    Op.Builder(opType = "DiagPart", name = name)
        .addInput(input)
        .build().outputs(0)
  }

  /** $OpDocMathMatrixDiag
    *
    * @group MathOps
    * @param  diagonal Rank-`K` input tensor, where `K >= 1`.
    * @param  name     Name for the created op.
    * @return Created op output with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its last
    *         dimension duplicated.
    */
  def matrixDiag(diagonal: Output, name: String = "MatrixDiag"): Output = {
    Op.Builder(opType = "MatrixDiag", name = name)
        .addInput(diagonal)
        .build().outputs(0)
  }

  /** $OpDocMathMatrixSetDiag
    *
    * @group MathOps
    * @param  input    Rank-`K+1` tensor, where `K >= 2`.
    * @param  diagonal Rank-`K` tensor, where `K >= 1`.
    * @param  name     Name for the created op.
    * @return Created op output with rank equal to `K + 1` and shape equal to the shape of `input`.
    */
  def matrixSetDiag(input: Output, diagonal: Output, name: String = "MatrixSetDiag"): Output = {
    Op.Builder(opType = "MatrixSetDiag", name = name)
        .addInput(input)
        .addInput(diagonal)
        .build().outputs(0)
  }

  /** $OpDocMathMatrixDiagPart
    *
    * @group MathOps
    * @param  input Rank-`K` tensor, where `K >= 2`.
    * @param  name  Name for the created op.
    * @return Created op output containing the diagonal(s) and having shape equal to
    *         `input.shape[:-2] + [min(input.shape[-2:])]`.
    */
  def matrixDiagPart(input: Output, name: String = "MatrixDiagPart"): Output = {
    Op.Builder(opType = "MatrixDiagPart", name = name)
        .addInput(input)
        .build().outputs(0)
  }

  /** $OpDocMathMatrixBandPart
    *
    * @group MathOps
    * @param  input             Input tensor.
    * @param  numSubDiagonals   Scalar `INT64` tensor that contains the number of sub-diagonals to keep. If negative,
    *                           the entire lower triangle is kept.
    * @param  numSuperDiagonals Scalar `INT64` tensor that contains the number of super-diagonals to keep. If negative,
    *                           the entire upper triangle is kept.
    * @param  name              Name for the created op.
    */
  def matrixBandPart(
      input: Output, numSubDiagonals: Output, numSuperDiagonals: Output, name: String = "MatrixBandPart"): Output = {
    if(!numSubDiagonals.dataType.isInteger)
      throw new IllegalArgumentException(s"'numSubDiagonals' must be integer, but was ${numSubDiagonals.dataType}.")
    if(!numSuperDiagonals.dataType.isInteger)
      throw new IllegalArgumentException(s"'numSuperDiagonals' must be integer, but was ${numSuperDiagonals.dataType}.")

    Op.Builder(opType = "MatrixBandPart", name = name)
        .addInput(input)
        .addInput(Cast.cast(numSubDiagonals, INT64))
        .addInput(Cast.cast(numSuperDiagonals, INT64))
        .build().outputs(0)
  }

  /** $OpDocMathTrace
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def trace(input: Output, name: String = "Trace"): Output = {
    Op.createWithNameScope(name) {
      sum(matrixDiagPart(input), axes = -1)
    }
  }

  /** $OpDocMathScalarMul
    *
    * @group MathOps
    * @param  scalar Scalar tensor.
    * @param  tensor Tensor to multiply the scalar tensor with.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  def scalarMul[T: OutputOps](scalar: Output, tensor: T, name: String = "ScalarMul"): T = {
    Op.createWithNameScope(name) {
      implicitly[OutputOps[T]].applyUnary(tensor, o => multiply(scalar, o))
    }
  }

  /** $OpDocMathMatmul
    *
    * @group MathOps
    * @param  a          First input tensor with data type one of: `BFLOAT16`, `FLOAT16`, `FLOAT32`, `FLOAT64`,
    *                    `INT32`, `COMPLEX64`, `COMPLEX128`.
    * @param  b          Second input tensor with data type one of: `BFLOAT16`, `FLOAT16`, `FLOAT32`, `FLOAT64`,
    *                    `INT32`, `COMPLEX64`, `COMPLEX128`.
    * @param  transposeA If `true`, `a` is transposed before the multiplication.
    * @param  transposeB If `true`, `b` is transposed before the multiplication.
    * @param  conjugateA If `true`, `a` is conjugated before the multiplication.
    * @param  conjugateB If `true`, `b` is conjugated before the multiplication.
    * @param  aIsSparse  If `true`, `a` is treated as a sparse matrix (i.e., it is assumed it contains many zeros).
    * @param  bIsSparse  If `true`, `b` is treated as a sparse matrix (i.e., it is assumed it contains many zeros).
    * @param  name       Name for the created op.
    * @return Created op output that has the same data type as `a` and `b` and where each inner-most matrix is the
    *         product of the corresponding matrices in `a` and `b`.
    */
  def matmul(
      a: Output, b: Output, transposeA: Boolean = false, transposeB: Boolean = false, conjugateA: Boolean = false,
      conjugateB: Boolean = false, aIsSparse: Boolean = false, bIsSparse: Boolean = false,
      name: String = "MatMul"): Output = {
    val (cA, cB) = castArgs(a, b)
    val sparseMatMulDataTypes = Set[DataType](BFLOAT16, FLOAT32)
    if (!aIsSparse && !bIsSparse && (cA.rank == -1 || cA.rank > 2) && (cB.rank == -1 || cB.rank > 2)) {
      // "BatchMatMul" does not support transpose, so we conjugate the matrix and use adjoint instead.
      // The "conj" op is a no-op for real matrices.
      val (x, adjointX) = transposeConjugateToAdjoint(cA, transposeA, conjugateA)
      val (y, adjointY) = transposeConjugateToAdjoint(cB, transposeB, conjugateB)
      Op.Builder(opType = "BatchMatMul", name = name)
          .addInput(x)
          .addInput(y)
          .setAttribute("adj_x", adjointX)
          .setAttribute("adj_y", adjointY)
          .build().outputs(0)
    } else if (cA.dataType == BFLOAT16 || cB.dataType == BFLOAT16 || // "MatMul" does not currently support this type.
        ((aIsSparse || bIsSparse) &&
            sparseMatMulDataTypes.contains(cA.dataType) &&
            sparseMatMulDataTypes.contains(cB.dataType))) {
      val (x, transposeX) = transposeConjugateToTranspose(cA, transposeA, conjugateA)
      val (y, transposeY) = transposeConjugateToTranspose(cB, transposeB, conjugateB)
      Op.Builder(opType = "SparseMatMul", name = name)
          .addInput(x)
          .addInput(y)
          .setAttribute("transpose_a", transposeX)
          .setAttribute("transpose_b", transposeY)
          .setAttribute("a_is_sparse", aIsSparse)
          .setAttribute("b_is_sparse", bIsSparse)
          .build().outputs(0)
    } else {
      val (x, transposeX) = transposeConjugateToTranspose(cA, transposeA, conjugateA)
      val (y, transposeY) = transposeConjugateToTranspose(cB, transposeB, conjugateB)
      Op.Builder(opType = "MatMul", name = name)
          .addInput(x)
          .addInput(y)
          .setAttribute("transpose_a", transposeX)
          .setAttribute("transpose_b", transposeY)
          .build().outputs(0)
    }
  }

  private[this] def transposeConjugateToAdjoint(
      tensor: Output, transpose: Boolean, conj: Boolean): (Output, Boolean) = {
    (transpose, conj) match {
      case (false, false) => (tensor, false)
      case (false, true) => (conjugate(tensor), false)
      case (true, false) => (conjugate(tensor), true)
      case (true, true) => (tensor, true)
    }
  }

  private[this] def transposeConjugateToTranspose(
      tensor: Output, transpose: Boolean, conj: Boolean): (Output, Boolean) = {
    (transpose, conj) match {
      case (false, false) => (tensor, false)
      case (false, true) => (conjugate(tensor), false)
      case (true, false) => (tensor, true)
      case (true, true) => (conjugate(tensor), true)
    }
  }

  /** $OpDocMathCross
    *
    * @group MathOps
    * @param  a    First input tensor.
    * @param  b    Second input tensor.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def cross(a: Output, b: Output, name: String = "Cross"): Output = {
    val (cA, cB) = castArgs(a, b)
    Op.Builder(opType = "Cross", name = name)
        .addInput(cA)
        .addInput(cB)
        .build().outputs(0)
  }

  /** $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a       First tensor.
    * @param  b       Second tensor.
    * @param  numAxes Number of axes to contract.
    * @return Created op output.
    */
  def tensorDot(a: Output, b: Output, numAxes: Int): Output = {
    tensorDot(a, b, numAxes, "TensorDot")
  }

  /** $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a       First tensor.
    * @param  b       Second tensor.
    * @param  numAxes Number of axes to contract.
    * @param  name    Name for the created ops.
    * @return Created op output.
    */
  def tensorDot(a: Output, b: Output, numAxes: Int, name: String): Output = {
    if (numAxes < 1)
      throw InvalidArgumentException("'numAxes' must be at least 1.")
    if (a.rank == -1)
      throw InvalidArgumentException(
        "Cannot use 'tensorDot' with an unknown input tensor shape. Use 'tensorDotDynamic' instead.")
    tensorDot(a, b, a.rank - numAxes until a.rank, 0 until numAxes, name)
  }

  /** $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a     First tensor.
    * @param  b     Second tensor.
    * @param  axesA Axes to contract in `a`.
    * @param  axesB Axes to contract in `b`.
    * @return Created op output.
    */
  def tensorDot(a: Output, b: Output, axesA: Seq[Int], axesB: Seq[Int]): Output = {
    tensorDot(a, b, axesA, axesB, "TensorDot")
  }

  /** $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a     First tensor.
    * @param  b     Second tensor.
    * @param  axesA Axes to contract in `a`.
    * @param  axesB Axes to contract in `b`.
    * @param  name  Name for the created ops.
    * @return Created op output.
    */
  def tensorDot(a: Output, b: Output, axesA: Seq[Int], axesB: Seq[Int], name: String): Output = {
    if (axesA.lengthCompare(axesB.size) != 0)
      throw InvalidArgumentException(
        s"Different number of contraction axes for 'a' and 'b', ${axesA.size} != ${axesB.size}.")

    /** Helper method to perform transpose and reshape for the tensor contraction op. This method is helpful in reducing
      * `tensorDot` to `matmul` using the `transpose` and the `reshape` ops. The method takes a tensor and performs the
      * correct transpose and reshape operations for the provided indices. It returns the reshaped tensor as well as a
      * list of indices necessary to reshape the tensor back to its proper shape after the matrix multiplication.
      *
      * @param  a       Tensor being reshaped.
      * @param  axes    Sequence of unique indices of axes of `a`.
      * @param  flipped If `true`, the method assumes that `a` is the second argument in the contraction operation.
      * @return Tuple that contains: (i) the reshaped tensor `a` that allows contraction via `matmul`, (ii) an `INT32`
      *         tensor that contains the shape of the free axes, and (iii) a sequence of integers representing the
      *         inferred static shape of the free axes.
      */
    def tensorDotReshape(a: Output, axes: Seq[Int], flipped: Boolean = false): (Output, Output, Seq[Int]) = {
      if (a.shape.isFullyDefined) {
        val mappedAxes = axes.map(i => if (i >= 0) i else i + a.rank)
        val prodAxes = mappedAxes.map(a.shape(_)).product
        val free = (0 until a.rank).filter(!mappedAxes.contains(_))
        val freeAxes = free.map(a.shape(_))
        val prodFree = freeAxes.product
        val permutation = if (flipped) mappedAxes ++ free else free ++ mappedAxes
        val newShape = if (flipped) Shape(prodAxes, prodFree) else Shape(prodFree, prodAxes)
        val reshapedA = Basic.reshape(Basic.transpose(a, permutation), newShape)
        val freeAxesOutput = if (freeAxes.isEmpty) Basic.constant(Tensor(INT32)) else Basic.constant(freeAxes)
        (reshapedA, freeAxesOutput, freeAxes)
      } else {
        val (mappedAxes, freeAxesStatic) = {
          if (a.rank != -1) {
            val mappedAxes = axes.map(i => if (i >= 0) i else i + a.rank)
            val free = (0 until a.rank).filter(!mappedAxes.contains(_))
            val freeAxes = free.map(a.shape(_))
            (mappedAxes, freeAxes)
          } else {
            (axes, null)
          }
        }
        val shapeA = Basic.shape(a, INT32)
        val rankA = Basic.rank(a)
        var axesO = Basic.constant(mappedAxes, name = "Axes")
        axesO = ((axesO >= 0).cast(INT32) * axesO) + ((axesO < 0).cast(INT32) * (axesO + rankA))
        val (free, _) = Basic.listDiff(Math.range(0, rankA), axesO)
        val freeAxes = Basic.gather(shapeA, free)
        val axesAxes = Basic.gather(shapeA, axesO)
        val prodFree = freeAxes.prod()
        val prodAxes = axesAxes.prod()
        val (permutation, newShape) = {
          if (flipped) {
            val permutation = Basic.concatenate(Seq(axesO, free), 0)
            val newShape = Basic.stack(Seq(prodAxes, prodFree))
            (permutation, newShape)
          } else {
            val permutation = Basic.concatenate(Seq(free, axesO), 0)
            val newShape = Basic.stack(Seq(prodFree, prodAxes))
            (permutation, newShape)
          }
        }
        val reshapedA = Basic.reshape(Basic.transpose(a, permutation), newShape)
        (reshapedA, freeAxes, freeAxesStatic)
      }
    }

    Op.createWithNameScope(name, Set(a.op, b.op)) {
      val (reshapedA, freeA, freeAStatic) = tensorDotReshape(a, axesA)
      val (reshapedB, freeB, freeBStatic) = tensorDotReshape(b, axesB, flipped = true)
      val abMatmul = matmul(reshapedA, reshapedB)
      val reshaped = Basic.reshape(abMatmul, Basic.concatenate(Seq(freeA, freeB), 0))
      if (freeAStatic != null && freeBStatic != null)
        reshaped.setShape(Shape.fromSeq(freeAStatic ++ freeBStatic))
      reshaped
    }
  }

  /** Dynamic version (i.e., where `numAxes` may be a symbolic tensor) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a       First tensor.
    * @param  b       Second tensor.
    * @param  numAxes Number of axes to contract.
    * @return Created op output.
    */
  def tensorDotDynamic(a: Output, b: Output, numAxes: Output): Output = {
    tensorDotDynamic(a, b, numAxes, "TensorDot")
  }

  /** Dynamic version (i.e., where `numAxes` may be a symbolic tensor) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a       First tensor.
    * @param  b       Second tensor.
    * @param  numAxes Number of axes to contract.
    * @param  name    Name for the created ops.
    * @return Created op output.
    */
  def tensorDotDynamic(a: Output, b: Output, numAxes: Output, name: String): Output = {
    if (numAxes.rank != 0)
      throw InvalidArgumentException("'numAxes' must be a scalar.")
    tensorDotDynamic(a, b, range(a.rank - numAxes, a.rank), range(0, numAxes), name)
  }

  /** Dynamic version (i.e., where `axesA` and `axesB` may be symbolic tensors) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a     First tensor.
    * @param  b     Second tensor.
    * @param  axesA Axes to contract in `a`.
    * @param  axesB Axes to contract in `b`.
    * @return Created op output.
    */
  def tensorDotDynamic(a: Output, b: Output, axesA: Output, axesB: Output): Output = {
    tensorDotDynamic(a, b, axesA, axesB, "TensorDot")
  }

  /** Dynamic version (i.e., where `axesA` and `axesB` may be symbolic tensors) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a     First tensor.
    * @param  b     Second tensor.
    * @param  axesA Axes to contract in `a`.
    * @param  axesB Axes to contract in `b`.
    * @param  name  Name for the created ops.
    * @return Created op output.
    */
  def tensorDotDynamic(a: Output, b: Output, axesA: Output, axesB: Output, name: String = "TensorDot"): Output = {
    if (axesA.rank != 1)
      throw InvalidArgumentException("'axesA' must be a vector.")
    if (axesB.rank != 1)
      throw InvalidArgumentException("'axesB' must be a vector.")

    /** Helper method to perform transpose and reshape for the tensor contraction op. This method is helpful in reducing
      * `tensorDot` to `matmul` using the `transpose` and the `reshape` ops. The method takes a tensor and performs the
      * correct transpose and reshape operations for the provided indices. It returns the reshaped tensor as well as a
      * list of indices necessary to reshape the tensor back to its proper shape after the matrix multiplication.
      *
      * @param  a       Tensor being reshaped.
      * @param  axes    Sequence of unique indices of axes of `a`.
      * @param  flipped If `true`, the method assumes that `a` is the second argument in the contraction operation.
      * @return Tuple that contains: (i) the reshaped tensor `a` that allows contraction via `matmul`, and (ii) an
      *         `INT32` tensor that contains the shape of the free axes.
      */
    def tensorDotReshape(a: Output, axes: Output, flipped: Boolean = false): (Output, Output) = {
      val shapeA = Basic.shape(a)
      val rankA = Basic.rank(a)
      val mappedAxes = ((axes >= 0).cast(INT32) * axes) + ((axes < 0).cast(INT32) * (axes + rankA))
      val (free, _) = Basic.listDiff(Math.range(0, rankA), mappedAxes)
      val freeAxes = Basic.gather(shapeA, free)
      val axesAxes = Basic.gather(shapeA, mappedAxes)
      val prodFree = freeAxes.prod()
      val prodAxes = axesAxes.prod()
      val (permutation, newShape) = {
        if (flipped) {
          val permutation = Basic.concatenate(Seq(mappedAxes, free), 0)
          val newShape = Basic.stack(Seq(prodAxes, prodFree))
          (permutation, newShape)
        } else {
          val permutation = Basic.concatenate(Seq(free, mappedAxes), 0)
          val newShape = Basic.stack(Seq(prodFree, prodAxes))
          (permutation, newShape)
        }
      }
      val reshapedA = Basic.reshape(Basic.transpose(a, permutation), newShape)
      (reshapedA, freeAxes)
    }

    Op.createWithNameScope(name, Set(a.op, b.op)) {
      val (reshapedA, freeA) = tensorDotReshape(a, axesA)
      val (reshapedB, freeB) = tensorDotReshape(b, axesB, flipped = true)
      val abMatmul = matmul(reshapedA, reshapedB)
      Basic.reshape(abMatmul, Basic.concatenate(Seq(freeA, freeB), 0))
    }
  }

  //endregion Matrix Ops

  //region Complex Ops

  /** $OpDocMathComplex
    *
    * @group MathOps
    * @param  real Tensor containing the real component. Must have [[FLOAT32]] or [[FLOAT64]] data type.
    * @param  imag Tensor containing the imaginary component. Must have [[FLOAT32]] or [[FLOAT64]] data type.
    * @param  name Name for the created op.
    * @return Created op output with data type being either [[COMPLEX64]] or [[COMPLEX128]].
    */
  def complex(real: Output, imag: Output, name: String = "Complex"): Output = {
    val (cReal, cImag) = castArgs(real, imag)
    val outputDataType = (cReal.dataType, cImag.dataType) match {
      case (FLOAT32, FLOAT32) => COMPLEX64
      case (FLOAT64, FLOAT64) => COMPLEX128
      case _ => throw new IllegalArgumentException(
        s"'real' (dataType = ${real.dataType}) and 'imag' (dataType = ${imag.dataType}) must both have the same data " +
            s"type, which must be either 'FLOAT32' or 'FLOAT64'.")
    }
    Op.Builder(opType = "Complex", name = name)
        .addInput(cReal)
        .addInput(cImag)
        .setAttribute("Tout", outputDataType)
        .build().outputs(0)
  }

  /** $OpDocMathReal
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def real[T <: OutputLike : OutputOps](input: T, name: String = "Real"): T = {
    if (!input.dataType.isComplex) {
      input
    } else {
      implicitly[OutputOps[T]]
          .applyUnary(input, o =>
            Op.Builder(opType = "Real", name = name)
                .addInput(o)
                .setAttribute("Tout", o.dataType.real)
                .build().outputs(0))
    }
  }

  /** $OpDocMathImag
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def imag[T <: OutputLike : OutputOps](input: T, name: String = "Imag"): T = {
    if (!input.dataType.isComplex) {
      input
    } else {
      implicitly[OutputOps[T]]
          .applyUnary(input, o =>
            Op.Builder(opType = "Imag", name = name)
                .addInput(o)
                .setAttribute("Tout", o.dataType.real)
                .build().outputs(0))
    }
  }

  /** $OpDocMathAngle
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If the provided tensor is not numeric.
    */
  @throws[IllegalArgumentException]
  def angle[T <: OutputLike : OutputOps](input: T, name: String = "Angle"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(input, o => {
          if (o.dataType.isComplex) {
            Op.Builder(opType = "Angle", name = name)
                .addInput(o)
                .setAttribute("Tout", o.dataType.real)
                .build().outputs(0)
          } else if (o.dataType.isNumeric) {
            Basic.zerosLike(o)
          } else {
            throw new IllegalArgumentException("'angle' can only take numeric tensors as input.")
          }
        })
  }

  /** $OpDocMathConjugate
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If the provided tensor is not numeric.
    */
  @throws[IllegalArgumentException]
  def conjugate[T <: OutputLike : OutputOps](input: T, name: String = "Conjugate"): T = {
    implicitly[OutputOps[T]]
        .applyUnary(input, o => {
          if (o.dataType.isComplex) {
            Op.Builder(opType = "Conj", name = name)
                .addInput(o)
                .build().outputs(0)
          } else if (o.dataType.isNumeric) {
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

  //region Bucketization Ops

  /** $OpDocMathBucketize
    *
    * @group MathOps
    * @param  input      Numeric tensor to bucketize.
    * @param  boundaries Sorted sequence of `Float`s specifying the boundaries of the buckets.
    * @param  name       Name for the created op.
    * @return Created op output.
    */
  def bucketize(input: Output, boundaries: Seq[Float], name: String = "Bucketize"): Output = {
    Op.Builder(opType = "Bucketize", name = name)
        .addInput(input)
        .setAttribute("boundaries", boundaries.toArray)
        .build().outputs(0)
  }

  //endregion Bucketization Ops

  //region Other Ops

  /** $OpDocMathZerosFraction
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  name  Name for the created op.
    * @return Created op output, with `FLOAT32` data type.
    */
  def zerosFraction(input: Output, name: String = "ZerosFraction"): Output = {
    Op.createWithNameScope(name, Set(input.op)) {
      val zero = Basic.constant(0, input.dataType, name = "Zero")
      mean(Cast.cast(equal(input, zero), FLOAT32))
    }
  }

  // TODO: bessel_i0, bessel_i1, bessel_i0e, bessel_i1e.

  //endregion Other Ops
}

object Math extends Math {
  case class MathOps(output: Output) {
    //region Math Operators

    /** $OpDocMathNegate
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def unary_- : Output = negate

    /** $OpDocMathAdd
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def +(other: Output): Output = add(other)

    /** $OpDocMathSubtract
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def -(other: Output): Output = subtract(other)

    /** $OpDocMathMultiply
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def *(other: Output): Output = multiply(other)

    private[this] def divHelper(x: Output, y: Output): Output = {
      if (x.dataType.isFloatingPoint || x.dataType.isComplex || y.dataType.isFloatingPoint || y.dataType.isComplex)
        Math.divide(x, y)
      else
        Math.truncateDivide(x, y)
    }

    /** $OpDocMathDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def /(other: Output): Output = divHelper(output, other)

    /** $OpDocMathMod
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def %(other: Output): Output = mod(other)

    /** $OpDocMathPow
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def **(other: Output): Output = pow(other)

    /** $OpDocMathPow
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def ^(other: Output): Output = pow(other)

    /** $OpDocMathLogicalNot
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def unary_! : Output = logicalNot

    /** $OpDocMathLogicalAnd
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def &&(other: Output): Output = logicalAnd(other)

    /** $OpDocMathLogicalOr
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def ||(other: Output): Output = logicalOr(other)

    /** $OpDocMathEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def ==(other: Output): Output = equal(other)

    /** $OpDocMathNotEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def !=(other: Output): Output = notEqual(other)

    /** $OpDocMathLess
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def <(other: Output): Output = less(other)

    /** $OpDocMathLessEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def <=(other: Output): Output = lessEqual(other)

    /** $OpDocMathGreater
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def >(other: Output): Output = greater(other)

    /** $OpDocMathGreaterEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def >=(other: Output): Output = greaterEqual(other)

    //endregion Math Operators

    //region Math Unary Ops

    /** $OpDocMathAbs
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def abs: Output = Math.abs(output)

    /** $OpDocMathNegate
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def negate: Output = Math.negate(output)

    /** $OpDocMathReciprocal
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def reciprocal: Output = Math.reciprocal(output)

    /** $OpDocMathSquare
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def square: Output = Math.square(output)

    /** $OpDocMathSqrt
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sqrt: Output = Math.sqrt(output)

    /** $OpDocMathRsqrt
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def rsqrt: Output = Math.rsqrt(output)

    /** $OpDocMathExp
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def exp: Output = Math.exp(output)

    /** $OpDocMathExpm1
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def expm1: Output = Math.expm1(output)

    /** $OpDocMathLog
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def log: Output = Math.log(output)

    /** $OpDocMathLog1p
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def log1p: Output = Math.log1p(output)

    /** $OpDocMathSin
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sin: Output = Math.sin(output)

    /** $OpDocMathCos
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def cos: Output = Math.cos(output)

    /** $OpDocMathTan
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def tan: Output = Math.tan(output)

    /** $OpDocMathAsin
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def asin: Output = Math.asin(output)

    /** $OpDocMathAcos
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def acos: Output = Math.acos(output)

    /** $OpDocMathAtan
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def atan: Output = Math.atan(output)

    /** $OpDocMathSinh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sinh: Output = Math.sinh(output)

    /** $OpDocMathCosh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def cosh: Output = Math.cosh(output)

    /** $OpDocMathTanh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def tanh: Output = Math.tanh(output)

    /** $OpDocMathAsinh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def asinh: Output = Math.asinh(output)

    /** $OpDocMathAcosh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def acosh: Output = Math.acosh(output)

    /** $OpDocMathAtanh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def atanh: Output = Math.atanh(output)

    /** $OpDocMathLogGamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logGamma: Output = Math.logGamma(output)

    /** $OpDocMathDigamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def digamma: Output = Math.digamma(output)

    /** $OpDocMathErf
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def erf: Output = Math.erf(output)

    /** $OpDocMathErfc
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def erc: Output = Math.erfc(output)

    /** $OpDocMathSigmoid
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sigmoid: Output = Math.sigmoid(output)

    /** $OpDocMathLogSigmoid
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logSigmoid: Output = Math.logSigmoid(output)

    /** $OpDocMathSign
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sign: Output = Math.sign(output)

    /** $OpDocMathRound
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def round: Output = Math.round(output)

    /** $OpDocMathRoundInt
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def roundInt: Output = Math.roundInt(output)

    /** $OpDocMathFloor
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def floor: Output = Math.floor(output)

    /** $OpDocMathCeil
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def ceil: Output = Math.ceil(output)

    /** $OpDocMathIsNaN
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def isNaN: Output = Math.isNaN(output)

    /** $OpDocMathIsInf
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def isInf: Output = Math.isInf(output)

    /** $OpDocMathIsFinite
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def isFinite: Output = Math.isFinite(output)

    //endregion Math Unary Ops

    //region Math Binary Ops

    /** $OpDocMathAdd
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def add(other: Output): Output = Math.add(output, other)

    /** $OpDocMathSubtract
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def subtract(other: Output): Output = Math.subtract(output, other)

    /** $OpDocMathMultiply
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def multiply(other: Output): Output = Math.multiply(output, other)

    /** $OpDocMathDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def divide(other: Output): Output = Math.divide(output, other)

    /** $OpDocMathFloorDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    @deprecated("Use `truncateDivide` instead.", "0.1")
    def floorDivide(other: Output): Output = Math.floorDivide(output, other)

    /** $OpDocMathTruncateDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def truncateDivide(other: Output): Output = Math.truncateDivide(output, other)

    /** $OpDocMathRealDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def realDivide(other: Output): Output = Math.realDivide(output, other)

    /** $OpDocMathSquaredDifference
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def squaredDifference(other: Output): Output = Math.squaredDifference(output, other)

    /** $OpDocMathMod
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def mod(other: Output): Output = Math.mod(output, other)

    /** $OpDocMathFloorMod
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def floorMod(other: Output): Output = Math.floorMod(output, other)

    /** $OpDocMathTruncateMod
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def truncateMod(other: Output): Output = Math.truncateMod(output, other)

    /** $OpDocMathPow
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def pow(other: Output): Output = Math.pow(output, other)

    /** $OpDocMathIgammac
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def igammac(other: Output): Output = Math.igammac(output, other)

    /** $OpDocMathIgamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def igamma(other: Output): Output = Math.igamma(output, other)

    /** $OpDocMathZeta
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def zeta(other: Output): Output = Math.zeta(output, other)

    /** $OpDocMathPolygamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def polygamma(other: Output): Output = Math.polygamma(output, other)

    /** $OpDocMathAtan2
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def atan2(other: Output): Output = Math.atan2(output, other)

    /** $OpDocMathMinimum
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def minimum(other: Output): Output = Math.minimum(output, other)

    /** $OpDocMathMaximum
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def maximum(other: Output): Output = Math.maximum(output, other)

    //endregion Math Binary Ops

    //region Math Logical Ops

    /** $OpDocMathLogicalNot
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logicalNot: Output = Math.logicalNot(output)

    /** $OpDocMathLogicalAnd
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logicalAnd(other: Output): Output = Math.logicalAnd(output, other)

    /** $OpDocMathLogicalOr
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logicalOr(other: Output): Output = Math.logicalOr(output, other)

    /** $OpDocMathLogicalXOr
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logicalXOr(other: Output): Output = Math.logicalXOr(output, other)

    //endregion Math Logical Ops

    //region Math Comparison Ops

    /** $OpDocMathEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def equal(other: Output): Output = Math.equal(output, other)

    /** $OpDocMathNotEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def notEqual(other: Output): Output = Math.notEqual(output, other)

    /** $OpDocMathApproximatelyEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def approximatelyEqual(other: Output): Output = Math.approximatelyEqual(output, other)

    /** $OpDocMathLess
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def less(other: Output): Output = Math.less(output, other)

    /** $OpDocMathLessEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def lessEqual(other: Output): Output = Math.lessEqual(output, other)

    /** $OpDocMathGreater
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def greater(other: Output): Output = Math.greater(output, other)

    /** $OpDocMathGreaterEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def greaterEqual(other: Output): Output = Math.greaterEqual(output, other)

    //endregion Math Comparison Ops

    //region Math Reduction Ops

    /** $OpDocMathSum
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def sum(axes: Output = null, keepDims: Boolean = false): Output = Math.sum(output, axes, keepDims)

    /** $OpDocMathMean
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def mean(axes: Output = null, keepDims: Boolean = false): Output = Math.mean(output, axes, keepDims)

    /** $OpDocMathProd
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def prod(axes: Output = null, keepDims: Boolean = false): Output = Math.prod(output, axes, keepDims)

    /** $OpDocMathMin
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def min(axes: Output = null, keepDims: Boolean = false): Output = Math.min(output, axes, keepDims)

    /** $OpDocMathMax
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def max(axes: Output = null, keepDims: Boolean = false): Output = Math.max(output, axes, keepDims)

    /** $OpDocMathAll
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def all(axes: Output = null, keepDims: Boolean = false): Output = Math.all(output, axes, keepDims)

    /** $OpDocMathAny
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def any(axes: Output = null, keepDims: Boolean = false): Output = Math.any(output, axes, keepDims)

    /** $OpDocMathLogSumExp
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def logSumExp(axes: Output = null, keepDims: Boolean = false): Output = Math.logSumExp(output, axes, keepDims)

    /** $OpDocMathCountNonZero
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def countNonZero(axes: Output = null, keepDims: Boolean = false): Output = Math.countNonZero(output, axes, keepDims)

    //endregion Math Reduction Ops

    /** $OpDocMathArgmax
      *
      * @group MathOps
      * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  outputDataType Data type for the output tensor. Must be `INT32` or `INT64`.
      * @return Result as a new tensor.
      */
    def argmax(axes: Output = 0, outputDataType: DataType = INT64): Output = Math.argmax(output, axes, outputDataType)

    /** $OpDocMathArgmin
      *
      * @group MathOps
      * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  outputDataType Data type for the output tensor. Must be `INT32` or `INT64`.
      * @return Result as a new tensor.
      */
    def argmin(axes: Output = 0, outputDataType: DataType = INT64): Output = Math.argmin(output, axes, outputDataType)

    /** $OpDocMathBinCount
      *
      * @group MathOps
      * @param  weights   If not `null`, this tensor must have the same shape as `input`. For each value in `input`, the
      *                   corresponding bin count will be incremented by the corresponding weight instead of `1`.
      * @param  minLength If not `null`, this ensures the output has length at least `minLength`, padding with zeros at
      *                   the end, if necessary.
      * @param  maxLength If not `null`, this skips values in `input` that are equal or greater than `maxLength`,
      *                   ensuring that the output has length at most `maxLength`.
      * @param  dataType  If `weights` is `null`, this determines the data type used for the output tensor (i.e., the
      *                   tensor containing the bin counts).
      * @return Result as a new tensor.
      */
    def binCount(
        weights: Output = null, minLength: Output = null, maxLength: Output = null,
        dataType: DataType = INT32): Output = {
      Math.binCount(output, weights, minLength, maxLength, dataType)
    }

    /** $OpDocMathCumsum
      *
      * @group MathOps
      * @param  axis      [[INT32]] tensor containing the axis along which to perform the cumulative sum.
      * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
      * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
      * @return Result as a new tensor.
      */
    def cumsum(axis: Output = 0, exclusive: Boolean = false, reverse: Boolean = false): Output = {
      Math.cumsum(output, axis, exclusive, reverse)
    }

    /** $OpDocMathCumprod
      *
      * @group MathOps
      * @param  axis      [[INT32]] tensor containing the axis along which to perform the cumulative product.
      * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative product.
      * @param  reverse   Boolean value indicating whether to perform a reverse cumulative product.
      * @return Result as a new tensor.
      */
    def cumprod(axis: Output = 0, exclusive: Boolean = false, reverse: Boolean = false): Output = {
      Math.cumprod(output, axis, exclusive, reverse)
    }

    //region Math Segment Ops

    /** $OpDocMathSegmentSum
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentSum(segmentIndices: Output): Output = Math.segmentSum(output, segmentIndices)

    /** $OpDocMathSegmentMean
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentMean(segmentIndices: Output): Output = Math.segmentMean(output, segmentIndices)

    /** $OpDocMathSegmentProd
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentProd(segmentIndices: Output): Output = Math.segmentProd(output, segmentIndices)

    /** $OpDocMathSegmentMin
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentMin(segmentIndices: Output): Output = Math.segmentMin(output, segmentIndices)

    /** $OpDocMathSegmentMax
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentMax(segmentIndices: Output): Output = Math.segmentMax(output, segmentIndices)

    /** $OpDocMathUnsortedSegmentSum
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
      * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
      * @return Result as a new tensor.
      */
    def unsortedSegmentSum(segmentIndices: Output, segmentsNumber: Output): Output = {
      Math.unsortedSegmentSum(output, segmentIndices, segmentsNumber)
    }

    /** $OpDocMathUnsortedSegmentMean
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
      * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
      * @return Result as a new tensor.
      */
    def unsortedSegmentMean(segmentIndices: Output, segmentsNumber: Output): Output = {
      Math.unsortedSegmentMean(output, segmentIndices, segmentsNumber)
    }

    /** $OpDocMathUnsortedSegmentProd
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
      * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
      * @return Result as a new tensor.
      */
    def unsortedSegmentProd(segmentIndices: Output, segmentsNumber: Output): Output = {
      Math.unsortedSegmentProd(output, segmentIndices, segmentsNumber)
    }

    /** $OpDocMathUnsortedSegmentMin
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
      * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
      * @return Result as a new tensor.
      */
    def unsortedSegmentMin(segmentIndices: Output, segmentsNumber: Output): Output = {
      Math.unsortedSegmentMin(output, segmentIndices, segmentsNumber)
    }

    /** $OpDocMathUnsortedSegmentMax
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
      * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
      * @return Result as a new tensor.
      */
    def unsortedSegmentMax(segmentIndices: Output, segmentsNumber: Output): Output = {
      Math.unsortedSegmentMax(output, segmentIndices, segmentsNumber)
    }

    /** $OpDocMathUnsortedSegmentSqrtN
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
      * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
      * @return Result as a new tensor.
      */
    def unsortedSegmentSqrtN(segmentIndices: Output, segmentsNumber: Output): Output = {
      Math.unsortedSegmentSqrtN(output, segmentIndices, segmentsNumber)
    }

    /** $OpDocMathSparseSegmentSum
      *
      * @group MathOps
      *
      * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @param  numSegments    Optional `INT32` scalar indicating the size of the output tensor.
      * @return Result as a new tensor.
      */
    def sparseSegmentSum(indices: Output, segmentIndices: Output, numSegments: Output = null): Output = {
      Math.sparseSegmentSum(output, indices, segmentIndices, numSegments)
    }

    /** $OpDocMathSparseSegmentMean
      *
      * @group MathOps
      *
      * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @param  numSegments    Optional `INT32` scalar indicating the size of the output tensor.
      * @return Result as a new tensor.
      */
    def sparseSegmentMean(indices: Output, segmentIndices: Output, numSegments: Output = null): Output = {
      Math.sparseSegmentMean(output, indices, segmentIndices, numSegments)
    }

    /** $OpDocMathSparseSegmentSumSqrtN
      *
      * @group MathOps
      *
      * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @param  numSegments    Optional `INT32` scalar indicating the size of the output tensor.
      * @return Result as a new tensor.
      */
    def sparseSegmentSumSqrtN(indices: Output, segmentIndices: Output, numSegments: Output = null): Output = {
      Math.sparseSegmentSumSqrtN(output, indices, segmentIndices, numSegments)
    }

    //endregion Math Segment Ops

    //region Math Matrix Ops

    /** $OpDocMathDiag
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def diag: Output = Math.diag(output)

    /** $OpDocMathDiagPart
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def diagPart: Output = Math.diagPart(output)

    /** $OpDocMathMatrixDiag
      *
      * @group MathOps
      *
      * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its
      *         last dimension duplicated.
      */
    def matrixDiag: Output = Math.matrixDiag(output)

    /** $OpDocMathMatrixSetDiag
      *
      * @group MathOps
      *
      * @param  diagonal Rank-`K` tensor, where `K >= 1`.
      * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `input`.
      */
    def matrixSetDiag(diagonal: Output): Output = Math.matrixSetDiag(output, diagonal)

    /** $OpDocMathMatrixDiagPart
      *
      * @group MathOps
      *
      * @return Result as a new tensor containing the diagonal(s) and having shape equal to
      *         `input.shape[:-2] + [min(input.shape[-2:])]`.
      */
    def matrixDiagPart: Output = Math.matrixDiagPart(output)

    /** $OpDocMathMatrixBandPart
      *
      * @group MathOps
      *
      * @param  numSubDiagonals   Scalar `INT64` tensor that contains the number of sub-diagonals to keep. If negative,
      *                           the entire lower triangle is kept.
      * @param  numSuperDiagonals Scalar `INT64` tensor that contains the number of super-diagonals to keep. If negative,
      *                           the entire upper triangle is kept.
      * @return Result as a new tensor containing the expected banded tensor and has rank `K` and same shape as `input`.
      */
    def matrixBandPart(numSubDiagonals: Output, numSuperDiagonals: Output): Output = {
      Math.matrixBandPart(output, numSubDiagonals, numSuperDiagonals)
    }

    /** $OpDocMathTrace
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def trace: Output = Math.trace(output)

    /** $OpDocMathMatmul
      *
      * @group MathOps
      *
      * @param  other      Output to multiply with, with data type one of: `BFLOAT16`, `FLOAT16`, `FLOAT32`, `FLOAT64`,
      *                    `INT32`, `COMPLEX64`, `COMPLEX128`.
      * @param  transposeA If `true`, this tensor is transposed before the multiplication.
      * @param  transposeB If `true`, `other` is transposed before the multiplication.
      * @param  conjugateA If `true`, this tensor is conjugated before the multiplication.
      * @param  conjugateB If `true`, `other` is conjugated before the multiplication.
      * @param  aIsSparse  If `true`, this tensor is treated as a sparse matrix (i.e., it is assumed it contains many
      *                    zeros).
      * @param  bIsSparse  If `true`, `other` is treated as a sparse matrix (i.e., it is assumed it contains many
      *                    zeros).
      * @return Result as a new tensor.
      */
    def matmul(
        other: Output, transposeA: Boolean = false, transposeB: Boolean = false, conjugateA: Boolean = false,
        conjugateB: Boolean = false, aIsSparse: Boolean = false, bIsSparse: Boolean = false): Output = {
      Math.matmul(output, other, transposeA, transposeB, conjugateA, conjugateB, aIsSparse, bIsSparse)
    }

    /** $OpDocMathCross
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def cross(other: Output): Output = Math.cross(output, other)

    /** $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other   Tensor to contract with.
      * @param  numAxes Number of axes to contract.
      * @return Created op output.
      */
    def tensorDot(other: Output, numAxes: Int): Output = {
      Math.tensorDot(output, other, numAxes)
    }

    /** $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other   Tensor to contract with.
      * @param  numAxes Number of axes to contract.
      * @param  name    Name for the created ops.
      * @return Created op output.
      */
    def tensorDot(other: Output, numAxes: Int, name: String): Output = {
      Math.tensorDot(output, other, numAxes, name)
    }

    /** $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other     Tensor to contract with.
      * @param  axes      Axes to contract in this tensor.
      * @param  axesOther Axes to contract in `other`.
      * @return Created op output.
      */
    def tensorDot(other: Output, axes: Seq[Int], axesOther: Seq[Int]): Output = {
      Math.tensorDot(output, other, axes, axesOther)
    }

    /** $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other     Tensor to contract with.
      * @param  axes      Axes to contract in this tensor.
      * @param  axesOther Axes to contract in `other`.
      * @param  name      Name for the created ops.
      * @return Created op output.
      */
    def tensorDot(other: Output, axes: Seq[Int], axesOther: Seq[Int], name: String): Output = {
      Math.tensorDot(output, other, axes, axesOther, name)
    }

    /** Dynamic version (i.e., where `numAxes` may be a symbolic tensor) of the `tensorDot` op.
      *
      * $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other   Tensor to contract with.
      * @param  numAxes Number of axes to contract.
      * @return Created op output.
      */
    def tensorDotDynamic(other: Output, numAxes: Output): Output = {
      Math.tensorDotDynamic(output, other, numAxes)
    }

    /** Dynamic version (i.e., where `numAxes` may be a symbolic tensor) of the `tensorDot` op.
      *
      * $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other   Tensor to contract with.
      * @param  numAxes Number of axes to contract.
      * @param  name    Name for the created ops.
      * @return Created op output.
      */
    def tensorDotDynamic(other: Output, numAxes: Output, name: String): Output = {
      Math.tensorDotDynamic(output, other, numAxes, name)
    }

    /** Dynamic version (i.e., where `axes` and `axesOther` may be symbolic tensors) of the `tensorDot` op.
      *
      * $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other     Tensor to contract with.
      * @param  axes      Axes to contract in this tensor.
      * @param  axesOther Axes to contract in `other`.
      * @return Created op output.
      */
    def tensorDotDynamic(other: Output, axes: Output, axesOther: Output): Output = {
      Math.tensorDotDynamic(output, other, axes, axesOther)
    }

    /** Dynamic version (i.e., where `axes` and `axesOther` may be symbolic tensors) of the `tensorDot` op.
      *
      * $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other     Tensor to contract with.
      * @param  axes      Axes to contract in this tensor.
      * @param  axesOther Axes to contract in `other`.
      * @param  name      Name for the created ops.
      * @return Created op output.
      */
    def tensorDotDynamic(other: Output, axes: Output, axesOther: Output, name: String): Output = {
      Math.tensorDotDynamic(output, other, axes, axesOther, name)
    }

    //endregion Math Matrix Ops

    //region Math Complex Ops

    /** $OpDocMathReal
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def real: Output = Math.real(output)

    /** $OpDocMathImag
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def imag: Output = Math.imag(output)

    /** $OpDocMathAngle
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def angle: Output = Math.angle(output)

    /** $OpDocMathConjugate
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def conjugate: Output = Math.conjugate(output)

    //endregion Math Complex Ops

    //region Math Quantization Ops

    // TODO: [OPS] quantization

    //endregion Math Quantization Ops

    //region Math Bucketization Ops

    /** $OpDocMathBucketize
      *
      * @group MathOps
      *
      * @param  boundaries Sorted sequence of `Float`s specifying the boundaries of the buckets.
      * @return Result as a new tensor.
      */
    def bucketize(boundaries: Seq[Float]): Output = Math.bucketize(output, boundaries)

    //endregion Math Bucketization Ops

    //region Math Other Ops

    /** $OpDocMathZerosFraction
      *
      * @group MathOps
      *
      * @return Result as a new tensor, with `FLOAT32` data type.
      */
    def zerosFraction: Output = Math.zerosFraction(output)

    //endregion Math Other Ops
  }

  private[ops] object Gradients {
    GradientsRegistry.registerNonDifferentiable("Range")
    GradientsRegistry.registerNonDifferentiable("LinSpace")
    GradientsRegistry.registerNonDifferentiable("IsNan")
    GradientsRegistry.registerNonDifferentiable("IsInf")
    GradientsRegistry.registerNonDifferentiable("IsFinite")
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

    GradientsRegistry.register("Select", selectGradient)
    GradientsRegistry.register("AddN", addNGradient)
    GradientsRegistry.register("AccumulateNV2", accumulateNGradient)
    GradientsRegistry.register("Abs", absGradient)
    GradientsRegistry.register("ComplexAbs", complexAbsGradient)
    GradientsRegistry.register("Neg", negateGradient)
    GradientsRegistry.register("Reciprocal", reciprocalGradient)
    GradientsRegistry.register("ReciprocalGrad", reciprocalHessian)
    GradientsRegistry.register("Square", squareGradient)
    GradientsRegistry.register("Sqrt", sqrtGradient)
    GradientsRegistry.register("SqrtGrad", sqrtHessian)
    GradientsRegistry.register("Rsqrt", rsqrtGradient)
    GradientsRegistry.register("RsqrtGrad", rsqrtHessian)
    GradientsRegistry.register("Exp", expGradient)
    GradientsRegistry.register("Expm1", expm1Gradient)
    GradientsRegistry.register("Log", logGradient)
    GradientsRegistry.register("Log1p", log1pGradient)
    GradientsRegistry.register("Sin", sinGradient)
    GradientsRegistry.register("Cos", cosGradient)
    GradientsRegistry.register("Tan", tanGradient)
    GradientsRegistry.register("Asin", asinGradient)
    GradientsRegistry.register("Acos", acosGradient)
    GradientsRegistry.register("Atan", atanGradient)
    GradientsRegistry.register("Sinh", sinhGradient)
    GradientsRegistry.register("Cosh", coshGradient)
    GradientsRegistry.register("Tanh", tanhGradient)
    GradientsRegistry.register("TanhGrad", tanhHessian)
    GradientsRegistry.register("Asinh", asinhGradient)
    GradientsRegistry.register("Acosh", acoshGradient)
    GradientsRegistry.register("Atanh", atanhGradient)
    GradientsRegistry.register("Lgamma", lgammaGradient)
    GradientsRegistry.register("Digamma", digammaGradient)
    GradientsRegistry.register("Erf", erfGradient)
    GradientsRegistry.register("Erfc", erfcGradient)
    GradientsRegistry.register("Sigmoid", sigmoidGradient)
    GradientsRegistry.register("SigmoidGrad", sigmoidHessian)
    GradientsRegistry.register("Sign", signGradient)
    GradientsRegistry.register("Round", roundGradient)
    GradientsRegistry.register("Rint", rintGradient)
    GradientsRegistry.register("Floor", floorGradient)
    GradientsRegistry.register("Ceil", ceilGradient)
    GradientsRegistry.register("Add", addGradient)
    GradientsRegistry.register("Sub", subGradient)
    GradientsRegistry.register("Mul", mulGradient)
    GradientsRegistry.register("Div", divGradient)
    GradientsRegistry.register("FloorDiv", floorDivGradient)
    GradientsRegistry.register("TruncateDiv", truncateDivGradient)
    GradientsRegistry.register("RealDiv", realDivGradient)
    GradientsRegistry.register("SquaredDifference", squaredDifferenceGradient)
    GradientsRegistry.register("Pow", powGradient)
    GradientsRegistry.register("Igammac", igammacGradient)
    GradientsRegistry.register("Igamma", igammaGradient)
    GradientsRegistry.register("Zeta", zetaGradient)
    GradientsRegistry.register("Polygamma", polygammaGradient)
    GradientsRegistry.register("Atan2", atan2Gradient)
    GradientsRegistry.register("Minimum", minimumGradient)
    GradientsRegistry.register("Maximum", maximumGradient)
    GradientsRegistry.register("Betainc", betaIncGradient)
    GradientsRegistry.register("Sum", sumGradient)
    GradientsRegistry.register("Mean", meanGradient)
    GradientsRegistry.register("Prod", prodGradient)
    GradientsRegistry.register("Min", minOrMaxGradient)
    GradientsRegistry.register("Max", minOrMaxGradient)
    GradientsRegistry.register("Cumsum", cumsumGradient)
    GradientsRegistry.register("Cumprod", cumprodGradient)
    GradientsRegistry.register("SegmentSum", segmentSumGradient)
    GradientsRegistry.register("SegmentMean", segmentMeanGradient)
    GradientsRegistry.register("SegmentMin", segmentMinOrMaxGradient)
    GradientsRegistry.register("SegmentMax", segmentMinOrMaxGradient)
    GradientsRegistry.register("UnsortedSegmentSum", unsortedSegmentSumGradient)
    GradientsRegistry.register("UnsortedSegmentProd", unsortedSegmentProdGradient)
    GradientsRegistry.register("UnsortedSegmentMin", unsortedSegmentMinOrMaxGradient)
    GradientsRegistry.register("UnsortedSegmentMax", unsortedSegmentMinOrMaxGradient)
    GradientsRegistry.register("SparseSegmentSum", sparseSegmentSumGradient)
    GradientsRegistry.register("SparseSegmentSumWithNumSegments", sparseSegmentSumWithNumSegmentsGradient)
    GradientsRegistry.register("SparseSegmentMean", sparseSegmentMeanGradient)
    GradientsRegistry.register("SparseSegmentMeanWithNumSegments", sparseSegmentMeanWithNumSegmentsGradient)
    GradientsRegistry.register("SparseSegmentSqrtN", sparseSegmentSumSqrtNGradient)
    GradientsRegistry.register("SparseSegmentSqrtNWithNumSegments", sparseSegmentSumSqrtNWithNumSegmentsGradient)
    GradientsRegistry.register("Diag", diagGradient)
    GradientsRegistry.register("DiagPart", diagPartGradient)
    GradientsRegistry.register("MatrixDiag", matrixDiagGradient)
    GradientsRegistry.register("MatrixSetDiag", matrixSetDiagGradient)
    GradientsRegistry.register("MatrixDiagPart", matrixDiagPartGradient)
    GradientsRegistry.register("MatrixBandPart", matrixBandPartGradient)
    GradientsRegistry.register("BatchMatMul", batchMatMulGradient)
    GradientsRegistry.register("MatMul", matMulGradient)
    GradientsRegistry.register("SparseMatMul", sparseMatMulGradient)
    GradientsRegistry.register("Cross", crossGradient)
    GradientsRegistry.register("Complex", complexGradient)
    GradientsRegistry.register("Real", realGradient)
    GradientsRegistry.register("Imag", imagGradient)
    GradientsRegistry.register("Conj", conjGradient)

    private[this] def selectGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val grad = outputGradients.head
      val c = op.inputs(0)
      val x = op.inputs(1)
      val zeros = Basic.zerosLike(x)
      Seq[OutputLike](null, select(c, grad, zeros), select(c, zeros, grad))
    }

    private[this] def addNGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq.fill(op.numInputs)(outputGradients.head)
    }

    private[this] def accumulateNGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq.fill(op.numInputs)(outputGradients.head)
    }

    private[this] def absGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(multiply(outputGradients.head.toOutput, sign(op.inputs(0))))
    }

    private[this] def complexAbsGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      Seq(multiply(complex(outputGradient, Basic.zerosLike(outputGradient)), sign(op.inputs(0))))
    }

    private[this] def negateGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(negate(outputGradients.head))
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

    private[this] def reciprocalGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      unaryGradientOp(op.outputs(0), outputGradients, opType = "ReciprocalGrad", name = "ReciprocalGradient")
    }

    private[this] def reciprocalHessian(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val a = op.inputs(0)
      val b = op.inputs(1)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val ca = conjugate(a)
        val cg = conjugate(outputGradient)
        val rg = unaryGradientOp(ca, outputGradients, opType = "ReciprocalGrad", name = "ReciprocalGradient")
        Seq(Basic.constant(-2, cg.dataType) * cg * b * ca, rg.head)
      }
    }

    private[this] def squareGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      // Using control dependencies to prevent 2*x from being computed too early.
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * (Basic.constant(2, x.dataType) * conjugate(x)))
      }
    }

    private[this] def sqrtGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      unaryGradientOp(op.outputs(0), outputGradients, opType = "SqrtGrad", name = "SqrtGradient")
    }

    private[this] def sqrtHessian(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val a = op.inputs(0)
      val y = op.outputs(0)
      val outputGradient = outputGradients.head.toOutput
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val ga = divide(outputGradient, a)
        Seq(negate(conjugate(ga)) * y, Basic.constant(0.5, ga.dataType) * ga)
      }
    }

    private[this] def rsqrtGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      unaryGradientOp(op.outputs(0), outputGradients, opType = "RsqrtGrad", name = "RSqrtGradient")
    }

    private[this] def rsqrtHessian(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val a = op.inputs(0)
      val b = op.inputs(1)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val ca = conjugate(a)
        val cg = conjugate(outputGradient)
        val rg = unaryGradientOp(ca, outputGradients, opType = "RsqrtGrad", name = "RSqrtGradient")
        Seq(Basic.constant(-1.5, cg.dataType) * cg * b * square(ca), rg.head)
      }
    }

    private[this] def expGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val y = op.outputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * conjugate(y))
      }
    }

    private[this] def expm1Gradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * exp(conjugate(x)))
      }
    }

    private[this] def logGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * reciprocal(conjugate(x)))
      }
    }

    private[this] def log1pGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * reciprocal(Basic.constant(1, x.dataType) + conjugate(x)))
      }
    }

    private[this] def sinGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * cos(conjugate(x)))
      }
    }

    private[this] def cosGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(negate(outputGradient) * sin(conjugate(x)))
      }
    }

    private[this] def tanGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * square(reciprocal(cos(conjugate(x)))))
      }
    }

    private[this] def asinGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * reciprocal(sqrt(Basic.constant(1, x.dataType) - square(conjugate(x)))))
      }
    }

    private[this] def acosGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(negate(outputGradient) * reciprocal(sqrt(Basic.constant(1, x.dataType) - square(conjugate(x)))))
      }
    }

    private[this] def atanGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * reciprocal(Basic.constant(1, x.dataType) + square(conjugate(x))))
      }
    }

    private[this] def sinhGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * cosh(conjugate(x)))
      }
    }

    private[this] def coshGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * sinh(conjugate(x)))
      }
    }

    private[this] def tanhGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      var y = op.outputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        y = conjugate(y)
        unaryGradientOp(y, outputGradients, opType = "TanhGrad", name = "TanhGradient")
      }
    }

    private[this] def tanhHessian(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val a = op.inputs(0)
      val b = op.inputs(1)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val ca = conjugate(a)
        val cb = conjugate(b)
        val rg = unaryGradientOp(ca, outputGradients, opType = "TanhGrad", name = "TanhGradient")
        Seq(Basic.constant(-2.0, outputGradient.dataType) * outputGradient * cb * ca, rg.head)
      }
    }

    private[this] def asinhGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val y = op.outputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient / cosh(conjugate(y)))
      }
    }

    private[this] def acoshGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val y = op.outputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient / sinh(conjugate(y)))
      }
    }

    private[this] def atanhGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * reciprocal(Basic.constant(1, x.dataType) - square(conjugate(x))))
      }
    }

    private[this] def lgammaGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * digamma(conjugate(x)))
      }
    }

    private[this] def digammaGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * polygamma(Basic.constant(1, x.dataType), conjugate(x)))
      }
    }

    private[this] def erfGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      val twoOverRootPi = Basic.constant(2.0 / math.sqrt(math.Pi), outputGradient.dataType)
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * twoOverRootPi * exp(negate(square(conjugate(x)))))
      }
    }

    private[this] def erfcGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val outputGradient = outputGradients.head
      val minusTwoOverRootPi = Basic.constant(-2.0 / math.sqrt(math.Pi), outputGradient.dataType)
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        Seq(outputGradient * minusTwoOverRootPi * exp(negate(square(conjugate(x)))))
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

    private[this] def sigmoidHessian(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val a = op.inputs(0)
      val b = op.inputs(1)
      val outputGradient = outputGradients.head
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val ca = conjugate(a)
        val cb = conjugate(b)
        val gb = outputGradient * cb
        val rg = unaryGradientOp(ca, outputGradients, opType = "SigmoidGrad", name = "SigmoidGradient")
        Seq(subtract(gb, Basic.constant(-2.0, outputGradient.dataType) * gb * ca), rg.head)
      }
    }

    private[this] def signGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(Basic.zerosLike(op.inputs(0)))
    }

    private[this] def roundGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(null)
    }

    private[this] def rintGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(null)
    }

    private[this] def floorGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(null)
    }

    private[this] def ceilGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(null)
    }

    /** Returns `true` if the shapes of `x`, `y`, and `gradient` are all fully specified (i.e., statically known)
      * and equal. */
    private[this] def shapeFullySpecifiedAndEqual(x: Output, y: Output, gradient: OutputLike): Boolean = {
      x.shape.isFullyDefined &&
          y.shape.isFullyDefined &&
          gradient.shape.isFullyDefined &&
          x.shape == y.shape &&
          x.shape == gradient.shape
    }

    private[this] def addGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val y = op.inputs(1)
      val outputGradient = outputGradients.head.toOutput
      if (shapeFullySpecifiedAndEqual(x, y, outputGradient)) {
        Seq(outputGradient, outputGradient)
      } else {
        val xShape = Basic.shape(x)
        val yShape = Basic.shape(y)
        val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
        Seq(
          Basic.reshape(sum(outputGradient, rx), xShape),
          Basic.reshape(sum(outputGradient, ry), yShape))
      }
    }

    private[this] def subGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val y = op.inputs(1)
      val outputGradient = outputGradients.head.toOutput
      if (shapeFullySpecifiedAndEqual(x, y, outputGradient)) {
        Seq(outputGradient, -outputGradient)
      } else {
        val xShape = Basic.shape(x)
        val yShape = Basic.shape(y)
        val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
        Seq(
          Basic.reshape(sum(outputGradient, rx), xShape),
          Basic.reshape(-sum(outputGradient, ry), yShape))
      }
    }

    private[this] def mulGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = conjugate(op.inputs(0))
      val y = conjugate(op.inputs(1))
      val outputGradient = outputGradients.head.toOutput
      if (shapeFullySpecifiedAndEqual(x, y, outputGradient) &&
          (outputGradient.dataType == INT32 || outputGradient.dataType == FLOAT32)) {
        Seq(outputGradient * y, outputGradient * x)
      } else {
        val xShape = Basic.shape(x)
        val yShape = Basic.shape(y)
        val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
        Seq(
          Basic.reshape(sum(multiply(outputGradient, y), rx), xShape),
          Basic.reshape(sum(multiply(x, outputGradient), ry), yShape))
      }
    }

    private[this] def divGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = conjugate(op.inputs(0))
      val y = conjugate(op.inputs(1))
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      val outputGradient = outputGradients.head.toOutput
      Seq(
        Basic.reshape(sum(divide(outputGradient, y), rx), xShape),
        Basic.reshape(sum(multiply(outputGradient, divide(divide(negate(x), y), y)), ry), yShape))
    }

    private[this] def floorDivGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(null, null)
    }

    private[this] def truncateDivGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(null, null)
    }

    private[this] def realDivGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = conjugate(op.inputs(0))
      val y = conjugate(op.inputs(1))
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      val outputGradient = outputGradients.head.toOutput
      Seq(
        Basic.reshape(sum(realDivide(outputGradient, y), rx), xShape),
        Basic.reshape(sum(multiply(outputGradient, realDivide(realDivide(negate(x), y), y)), ry), yShape))
    }

    private[this] def squaredDifferenceGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val y = op.inputs(1)
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      val outputGradient = outputGradients.head.toOutput
      val xGradient = Op.createWith(controlDependencies = Set(outputGradient.op)) {
        multiply(scalarMul(Basic.constant(2, outputGradient.dataType), outputGradient), subtract(x, y))
      }
      Seq(
        Basic.reshape(sum(xGradient, rx), xShape),
        Basic.reshape(sum(xGradient, ry), yShape))
    }

    private[this] def powGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = conjugate(op.inputs(0))
      val y = conjugate(op.inputs(1))
      val z = conjugate(op.outputs(0))
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      val outputGradient = outputGradients.head.toOutput
      // Avoid false singularity at x = 0.
      val logX = {
        if (x.dataType.isComplex) {
          // real(x) < 0 is fine for the complex case.
          select(notEqual(x, Basic.constant(0, x.dataType)), log(x), Basic.zerosLike(x))
        } else {
          // There's no sensible real value to return if x < 0, so we return 0.
          select(greater(x, Basic.constant(0, x.dataType)), log(x), Basic.zerosLike(x))
        }
      }
      Seq(
        Basic.reshape(sum(outputGradient * y * pow(x, subtract(y, Basic.constant(1, y.dataType))), rx), xShape),
        Basic.reshape(sum(outputGradient * z * logX, ry), yShape))
    }

    private[this] def igammacGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val igammaGradients = igammaGradient(op, outputGradients)
      Seq(negate(igammaGradients(0)), negate(igammaGradients(1)))
    }

    private[this] def igammaGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val a = op.inputs(0)
      val x = op.inputs(1)
      val aShape = Basic.shape(a)
      val xShape = Basic.shape(x)
      val (ra, rx) = Basic.broadcastGradientArguments(aShape, xShape)
      val outputGradient = outputGradients.head.toOutput
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val partialA = Op.Builder(opType = "IgammaGradA", name = "IGammaGradA")
            .addInput(a)
            .addInput(x)
            .build().outputs(0)
        // Perform operations in log space before summing, because Gamma(a) and Gamma'(a) can grow large.
        val partialX = exp(negate(x) + multiply(subtract(a, Basic.constant(1, a.dataType)), log(x)) - logGamma(a))
        Seq(
          Basic.reshape(sum(multiply(partialA, outputGradient), ra), aShape),
          Basic.reshape(sum(multiply(partialX, outputGradient), rx), xShape))
      }
    }

    private[this] def zetaGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      // TODO: [GRADIENTS] Mark the derivative w.r.t. x as not implemented somehow, or implement it.
      val outputGradient = outputGradients.head.toOutput
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val x = conjugate(op.inputs(0))
        val q = conjugate(op.inputs(1))
        val xShape = Basic.shape(x)
        val qShape = Basic.shape(q)
        val (_, rq) = Basic.broadcastGradientArguments(xShape, qShape)
        val partialQ = negate(x) * zeta(add(x, Basic.constant(1, x.dataType)), q)
        Seq(null, Basic.reshape(sum(multiply(partialQ, outputGradient), rq), qShape))
      }
    }

    private[this] def polygammaGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      // TODO: [GRADIENTS] Mark the derivative w.r.t. n as not implemented somehow, or implement it.
      val outputGradient = outputGradients.head.toOutput
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val n = conjugate(op.inputs(0))
        val x = conjugate(op.inputs(1))
        val nShape = Basic.shape(n)
        val xShape = Basic.shape(x)
        val (_, rx) = Basic.broadcastGradientArguments(nShape, xShape)
        val partialX = polygamma(add(n, Basic.constant(1, n.dataType)), x)
        Seq(null, Basic.reshape(sum(multiply(partialX, outputGradient), rx), xShape))
      }
    }

    private[this] def atan2Gradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val y = op.inputs(1)
      val outputGradient = outputGradients.head.toOutput
      Op.createWith(controlDependencies = Set(outputGradient.op)) {
        val gradientInverse = divide(outputGradient, add(square(x), square(y)))
        Seq(
          multiply(x, gradientInverse),
          multiply(negate(y), gradientInverse))
      }
    }

    private[this] def minimumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val y = op.inputs(1)
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val outputGradient = outputGradients.head.toOutput
      val zeros = Basic.zerosLike(outputGradient)
      val xMask = lessEqual(x, y)
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      val xGradient = select(xMask, outputGradient, zeros)
      val yGradient = select(xMask, zeros, outputGradient)
      Seq(
        Basic.reshape(sum(xGradient, rx), xShape),
        Basic.reshape(sum(yGradient, ry), yShape))
    }

    private[this] def maximumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val y = op.inputs(1)
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val outputGradient = outputGradients.head.toOutput
      val zeros = Basic.zerosLike(outputGradient)
      val xMask = greaterEqual(x, y)
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      val xGradient = select(xMask, outputGradient, zeros)
      val yGradient = select(xMask, outputGradient, zeros)
      Seq(
        Basic.reshape(sum(xGradient, rx), xShape),
        Basic.reshape(sum(yGradient, ry), yShape))
    }

    private[this] def betaIncGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      // TODO: [GRADIENTS] Mark the derivative w.r.t. a and b as not implemented somehow, or implement it.
      val a = conjugate(op.inputs(0))
      val b = conjugate(op.inputs(1))
      val x = conjugate(op.inputs(2))
      val aShape = Basic.shape(a)
      val xShape = Basic.shape(x)
      val outputGradient = outputGradients.head.toOutput
      val (_, rx) = Basic.broadcastGradientArguments(aShape, xShape)
      // Perform operations in log space before summing, because terms can grow large.
      val logBeta = logGamma(a) + logGamma(b) - logGamma(a + b)
      val one = Basic.constant(1, b.dataType)
      val partialX = exp(((b - 1) * log(one - x)) + ((a - one) * log(x)) - logBeta)
      Seq(null, null, Basic.reshape(sum(multiply(partialX, outputGradient), rx), xShape))
    }

    private[this] def safeShapeDiv(x: Output, y: Output): Output = {
      truncateDivide(x, maximum(y, Basic.constant(1, y.dataType)))
    }

    private[this] def sumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val input = op.inputs(0)
      val axes = op.inputs(1)
      val rank = input.shape.rank
      // Fast path for when reducing to a scalar and rank is known, which adds only reshape and tile ops (and possibly a
      // shape op too).
      if (rank == 0) {
        Seq(outputGradients.head, null)
      } else if (rank != -1
          && axes.op.opType == "Const"
          && Output.constantValue(axes).exists(a => a.toInt32.entriesIterator.toArray[Int].sameElements((0 until rank).toArray[Int]))) {
        // In this case the reduction was over all dimensions.
        var outputGradient = outputGradients.head.toOutput
        outputGradient = Basic.reshape(outputGradient, Shape(Array.fill(rank)(1)))
        val inputShape = {
          // If the shape is not fully defined but the rank is, we use the shape op.
          if (input.shape.isFullyDefined)
            input.shape.toOutput()
          else
            Basic.shape(input)
        }
        Seq(Basic.tile(outputGradient, inputShape), null)
      } else {
        val inputShape = Basic.shape(input)
        val outputShapeKeptDimensions = reducedShape(inputShape, axes)
        val tileScaling = safeShapeDiv(inputShape, outputShapeKeptDimensions)
        var outputGradient = outputGradients.head.toOutput
        outputGradient = Basic.reshape(outputGradient, outputShapeKeptDimensions)
        Seq(Basic.tile(outputGradient, tileScaling), null)
      }
    }

    private[this] def meanGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val sumGrad = sumGradient(op, outputGradients).head.toOutput
      val factor = {
        val inputSize = op.inputs(0).size
        val outputSize = op.outputs(0).size
        if (inputSize != -1 && outputSize != -1) {
          Basic.constant(inputSize / scala.math.max(outputSize, 1), sumGrad.dataType)
        } else {
          val inputShape = Basic.shape(op.inputs(0))
          val outputShape = Basic.shape(op.outputs(0))
          safeShapeDiv(prod(inputShape), prod(outputShape))
        }
      }
      Seq(divide(sumGrad, Cast.cast(factor, sumGrad.dataType)), null)
    }

    private[this] def prodGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      // The gradient can be expressed by dividing the product by each entry of the input tensor, but this approach
      // can't deal with zeros in the input. Here, we avoid this problem by composing the output as a product of two
      // cumulative product operations.
      val inputShape = Basic.shape(op.inputs(0))
      // Expand the gradient to the full input shape
      val outputShapeKeptDims = reducedShape(inputShape, op.inputs(1))
      val tileScaling = safeShapeDiv(inputShape, outputShapeKeptDims)
      var gradient = outputGradients.head.toOutput
      gradient = Basic.reshape(gradient, outputShapeKeptDims)
      gradient = Basic.tile(gradient, tileScaling)

      // Pack all reduced dimensions into a single one, so we can perform the cumulative product ops. If the reduction
      // dimensions list is empty, it defaults to FLOAT32 data type, so we need to cast here. We place all the
      // shape-related ops on the CPU to avoid copying back and forth, and since "listdiff" is a CPU-only op.
      val (permutation, reducedNum, otherNum) = Op.createWith(device = "/cpu:0") {
        val rank = Basic.rank(op.inputs(0))
        // Reshape the reduction indices for the case where the parameters is a scalar.
        val reductionIndices = floorMod(add(Basic.reshape(op.inputs(1), -1), rank), rank)
        val reduced = Cast.cast(reductionIndices, INT32)
        val indices = range(Basic.constant(0), rank)
        val (other, _) = Basic.listDiff(indices, reduced)
        (Basic.concatenate(Seq(reduced, other), 0),
            prod(Basic.gather(inputShape, reduced)),
            prod(Basic.gather(inputShape, other)))
      }

      val permuted = Basic.transpose(op.inputs(0), permutation)
      val permutedShape = Basic.shape(permuted)
      val reshaped = Basic.reshape(permuted, Basic.concatenate(Seq(reducedNum, otherNum)))

      // Calculate the product, leaving out the current entry.
      val left = cumprod(reshaped, axis = 0, exclusive = true)
      val right = cumprod(reshaped, axis = 0, exclusive = true, reverse = true)
      // For complex inputs, the gradient is in the conjugate direction.
      val y = Basic.reshape(multiply(Math.conjugate(left), Math.conjugate(right)), permutedShape)

      // Invert the transpose and reshape operations.
      val output = multiply(gradient, Basic.transpose(y, Basic.invertPermutation(permutation)))
      // Make sure to set the statically known shape information through a reshape.
      Seq(Basic.reshape(output, inputShape), null)
    }

    private[this] def minOrMaxGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val inputShape = Basic.shape(op.inputs(0))
      val outputShapeKeptDims = reducedShape(inputShape, op.inputs(1))
      val y = Basic.reshape(op.outputs(0), outputShapeKeptDims)
      var gradient = outputGradients.head.toOutput
      gradient = Basic.reshape(gradient, outputShapeKeptDims)

      // Compute the number of selected (maximum or minimum) elements in each reduction dimension. If there are multiple
      // minimum or maximum elements then the gradient will be divided among them.
      val indicators = Cast.cast(equal(y, op.inputs(0)), gradient.dataType)
      val numberOfSelected = Basic.reshape(sum(indicators, op.inputs(1)), outputShapeKeptDims)

      Seq(multiply(divide(indicators, numberOfSelected), gradient), null)
    }

    private[this] def cumsumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val axis = op.inputs(1)
      val exclusive = op.booleanAttribute("exclusive")
      val reverse = op.booleanAttribute("reverse")
      val outputGradient = outputGradients.head
      Seq(cumsum(outputGradient, axis, exclusive = exclusive, reverse = !reverse), null)
    }

    private[this] def cumprodGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val axis = op.inputs(1)
      val exclusive = op.booleanAttribute("exclusive")
      val reverse = op.booleanAttribute("reverse")
      val outputGradient = outputGradients.head
      // TODO: [GRADIENTS] !!! This fails when x contains 0 and should be fixed.
      val product = cumprod(x, axis, exclusive = exclusive, reverse = reverse)
      val result = cumsum(product * outputGradient, axis, exclusive = exclusive, reverse = !reverse)
      Seq(divide(result, x), null)
    }

    private[this] def segmentSumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      Seq(Basic.gather(outputGradient, op.inputs(1)), null)
    }

    private[this] def segmentMeanGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      val inputRank = Basic.rank(op.inputs(0))
      val onesShape = Basic.concatenate(Seq(
        Basic.shape(op.inputs(1)),
        Basic.fill(
          shape = Basic.expandDims(subtract(inputRank, Basic.constant(1, inputRank.dataType)), 0))(
          Basic.constant(1, inputRank.dataType))))
      val ones = Basic.fill(shape = onesShape)(Basic.constant(1, outputGradient.dataType))
      val scaledGradient = divide(outputGradient, segmentSum(ones, op.inputs(1)))
      Seq(Basic.gather(scaledGradient, op.inputs(1)), null)
    }

    private[this] def segmentMinOrMaxGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      // Get the number of selected (minimum or maximum) elements in each segment.
      val gatheredOutputs = Basic.gather(op.outputs(0), op.inputs(1))
      val isSelected = equal(op.inputs(0), gatheredOutputs)
      val numSelected = segmentSum(Cast.cast(isSelected, outputGradient.dataType), op.inputs(1))

      // Compute the gradient for each segment. The gradient for the ith segment is divided evenly among the selected
      // elements in that segment.
      val weightedGradients = divide(outputGradient, numSelected)
      val gatheredGradients = Basic.gather(weightedGradients, op.inputs(1))
      val zeros = Basic.zerosLike(gatheredGradients)

      Seq(select(isSelected, gatheredGradients, zeros), null)
    }

    private[this] def gatherDropNegatives(
        parameters: Output,
        indices: Output,
        zeroClippedIndices: Output = null,
        isPositive: Output = null
    ): (Output, Output, Output) = {
      val computedZeroClippedIndices = {
        if (zeroClippedIndices != null)
          zeroClippedIndices
        else
          Math.maximum(indices, Basic.zerosLike(indices))
      }
      val gathered = Basic.gather(parameters, zeroClippedIndices)
      val computedIsPositive = {
        if (isPositive != null) {
          isPositive
        } else {
          var isPositive = Math.greaterEqual(indices, 0)
          // `select` requires that the condition has the same shape as the other two arguments.
          val minusOne = Basic.constant(-1)
          (0 until (gathered.rank - isPositive.rank)).foreach(_ => {
            isPositive = Basic.expandDims(isPositive, minusOne)
          })
          Math.logicalAnd(isPositive, Basic.onesLike(gathered, dataType = BOOLEAN))
        }
      }
      (Math.select(computedIsPositive, gathered, Basic.zerosLike(gathered)),
          computedZeroClippedIndices,
          computedIsPositive)
    }

    private[this] def unsortedSegmentSumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      Seq(gatherDropNegatives(outputGradient, op.inputs(1))._1, null, null)
    }

    private[this] def unsortedSegmentProdGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      // This gradient can be expressed for each segment by dividing the segment's product by each element of the
      // segment input tensor, but this approach cannot deal with zeros in the input. Unlike `prod` we cannot use the
      // cumulative sum op here, as individual segments may have a different number of elements. Therefore, we consider
      // three cases:
      //
      //   1) A segment input contains no zeros and can safely be divided by the input tensor.
      //   2) A segment contains exactly one zero. In this case, the gradient of each input of the segment is zero,
      //      except for the 0-input. There the gradient is the product of the remaining segment entries.
      //   3) A segment contains at least two zeros. In this case, the gradient is zero for all segment inputs.

      var outputGradient = outputGradients.head.toOutput
      // Note that `unsortedSegmentSum` will filter out the negative indices, and so we do not need to do a `logicalAnd`
      // with `isPositive` here.
      val isZero = Math.equal(op.inputs(0), 0)
      val numZeros = Math.unsortedSegmentSum(Cast.cast(isZero, INT32), op.inputs(1), op.inputs(2))
      // Handle case 3 and set the gradient to 0 for segments with more than one 0 as input.
      outputGradient = Math.select(Math.greater(numZeros, 1), Basic.zerosLike(outputGradient), outputGradient)
      // Replace all zeros with ones and compute the `unsortedSegmentProd`.
      val nonZeroData = Math.select(isZero, Basic.onesLike(op.inputs(0)), op.inputs(0))
      val nonZeroProd = Math.unsortedSegmentProd(nonZeroData, op.inputs(1), op.inputs(2))
      // Clip the indices for the gather to be positive.
      val zeroClippedIndices = Math.maximum(op.inputs(1), Basic.zerosLike(op.inputs(1)))
      val gatheredProd = Basic.gather(op.outputs(0), zeroClippedIndices)
      val gatheredNonZeroProd = Basic.gather(nonZeroProd, zeroClippedIndices)
      // The following may contain NaN/Inf.
      val gatheredProdDivided = gatheredProd / op.inputs(0)
      // Now fetch the individual results for segments containing zero and those that do not. `isZero` will also fetch
      // results for entries with negative indices, but the following `gatherDropNegatives` sets the corresponding entry
      // in the gradient to zero for these.
      val partialDerivative = Math.select(isZero, gatheredNonZeroProd, gatheredProdDivided)
      val gatheredGradient = gatherDropNegatives(outputGradient, op.inputs(1), zeroClippedIndices)._1
      Seq(gatheredGradient * partialDerivative, null, null)
    }

    private[this] def unsortedSegmentMinOrMaxGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      // Get the number of selected (minimum or maximum) elements in each segment.
      val (gatheredOutputs, zeroClippedIndices, isPositive) = gatherDropNegatives(op.outputs(0), op.inputs(1))
      val isSelected = Math.logicalAnd(Math.equal(op.inputs(0), gatheredOutputs), isPositive)
      val numSelected = unsortedSegmentSum(Cast.cast(isSelected, outputGradient.dataType), op.inputs(1), op.inputs(2))
      // Compute the gradient for each segment. The gradient for the ith segment is divided evenly among the selected
      // elements in that segment.
      val weightedGradients = divide(outputGradient, numSelected)
      val (gatheredGradients, _, _) = gatherDropNegatives(weightedGradients, null, zeroClippedIndices, isPositive)
      val zeros = Basic.zerosLike(gatheredGradients)

      Seq(select(isSelected, gatheredGradients, zeros), null, null)
    }

    private[this] def sparseSegmentSumGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      val inputRows = Basic.shape(op.inputs(0))(0)
      Seq(unsortedSegmentSum(Basic.gather(outputGradient, op.inputs(2)), op.inputs(1), inputRows), null, null)
    }

    private[this] def sparseSegmentSumWithNumSegmentsGradient(
        op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      sparseSegmentSumGradient(op, outputGradients) :+ null
    }

    private[this] def sparseSegmentMeanGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      val inputRows = Basic.shape(op.inputs(0))(0)
      val gradient = Op.Builder(opType = "SparseSegmentMeanGrad", name = "SparseSegmentMeanGrad")
          .addInput(outputGradient)
          .addInput(op.inputs(1))
          .addInput(op.inputs(2))
          .addInput(inputRows)
          .build().outputs(0)
      Seq(gradient, null, null)
    }

    private[this] def sparseSegmentMeanWithNumSegmentsGradient(
        op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      sparseSegmentMeanGradient(op, outputGradients) :+ null
    }

    private[this] def sparseSegmentSumSqrtNGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      val inputRows = Basic.shape(op.inputs(0))(0)
      val gradient = Op.Builder(opType = "SparseSegmentSqrtNGrad", name = "SparseSegmentSumSqrtNGrad")
          .addInput(outputGradient)
          .addInput(op.inputs(1))
          .addInput(op.inputs(2))
          .addInput(inputRows)
          .build().outputs(0)
      Seq(gradient, null, null)
    }

    private[this] def sparseSegmentSumSqrtNWithNumSegmentsGradient(
        op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      sparseSegmentSumSqrtNGradient(op, outputGradients) :+ null
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
          Basic.constant(tensors.ops.Basic.stack((batchShape.asArray :+ matrixShape.asArray.min).map(Tensor(_))))
        } else {
          Op.colocateWith(Set(gradient.op), ignoreExisting = true) {
            val gradShape = Basic.shape(gradient)
            val gradRank = Basic.rank(gradient)
            val batchShape = Basic.slice(gradShape, 0, gradRank - 2)
            val matrixShape = Basic.slice(gradShape, gradRank - 2, 2)
            val minDim = min(matrixShape)
            Basic.concatenate(Seq(batchShape, minDim), 0)
          }
        }
      }
      val gradInput = matrixSetDiag(gradient, Basic.fill(shape = diagShape)(Tensor.zeros(gradient.dataType, Shape())))
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

    private[this] def batchMatMulGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val y = op.inputs(1)
      val adjointX = op.booleanAttribute("adj_x")
      val adjointY = op.booleanAttribute("adj_y")
      val outputGradient = outputGradients.head.toOutput
      (adjointX, adjointY) match {
        case (false, false) =>
          Seq[OutputLike](
            matmul(outputGradient, y, transposeA = false, transposeB = true, conjugateA = false, conjugateB = true),
            matmul(x, outputGradient, transposeA = true, transposeB = false, conjugateA = true, conjugateB = false))
        case (false, true) =>
          Seq[OutputLike](
            matmul(outputGradient, y, transposeA = false, transposeB = false, conjugateA = false, conjugateB = false),
            matmul(outputGradient, x, transposeA = true, transposeB = false, conjugateA = true, conjugateB = false))
        case (true, false) =>
          Seq[OutputLike](
            matmul(y, outputGradient, transposeA = false, transposeB = true, conjugateA = false, conjugateB = true),
            matmul(x, outputGradient, transposeA = false, transposeB = false, conjugateA = false, conjugateB = false))
        case (true, true) =>
          Seq[OutputLike](
            matmul(y, outputGradient, transposeA = true, transposeB = true, conjugateA = true, conjugateB = true),
            matmul(outputGradient, x, transposeA = true, transposeB = true, conjugateA = true, conjugateB = true))
      }
    }

    private[this] def matMulGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val a = conjugate(op.inputs(0))
      val b = conjugate(op.inputs(1))
      val transposeA = op.booleanAttribute("transpose_a")
      val transposeB = op.booleanAttribute("transpose_b")
      val outputGradient = outputGradients.head.toOutput
      (transposeA, transposeB) match {
        case (false, false) =>
          Seq[OutputLike](
            matmul(outputGradient, b, transposeA = false, transposeB = true, conjugateA = false, conjugateB = false),
            matmul(a, outputGradient, transposeA = true, transposeB = false, conjugateA = false, conjugateB = false))
        case (false, true) =>
          Seq[OutputLike](
            matmul(outputGradient, b, transposeA = false, transposeB = false, conjugateA = false, conjugateB = false),
            matmul(outputGradient, a, transposeA = true, transposeB = false, conjugateA = false, conjugateB = false))
        case (true, false) =>
          Seq[OutputLike](
            matmul(b, outputGradient, transposeA = false, transposeB = true, conjugateA = false, conjugateB = false),
            matmul(a, outputGradient, transposeA = false, transposeB = false, conjugateA = false, conjugateB = false))
        case (true, true) =>
          Seq[OutputLike](
            matmul(b, outputGradient, transposeA = true, transposeB = true, conjugateA = false, conjugateB = false),
            matmul(outputGradient, a, transposeA = true, transposeB = true, conjugateA = false, conjugateB = false))
      }
    }

    private[this] def sparseMatMulGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val a = op.inputs(0)
      val b = op.inputs(1)
      val transposeA = op.booleanAttribute("transpose_a")
      val transposeB = op.booleanAttribute("transpose_b")
      val outputGradient = outputGradients.head.toOutput
      val aIsSparse = op.booleanAttribute("a_is_sparse")
      val bIsSparse = op.booleanAttribute("b_is_sparse")
      // Use heuristic to figure out if the gradient may be sparse.
      val gradIsSparse = outputGradient.op.opType == "ReluGrad"

      def helper(
          a: Output, b: Output, dataType: DataType,
          tA: Boolean = false, tB: Boolean = false,
          sA: Boolean = false, sB: Boolean = false): Output = {
        Cast.cast(matmul(
          a = a,
          b = if (tB) Basic.transpose(b) else b,
          transposeA = tA,
          transposeB = false,
          conjugateA = false,
          conjugateB = false,
          aIsSparse = sA,
          bIsSparse = sB), dataType)
      }

      (transposeA, transposeB) match {
        case (false, false) =>
          Seq[OutputLike](
            helper(outputGradient, b, a.dataType, tA = false, tB = true, sA = gradIsSparse, sB = bIsSparse),
            helper(a, outputGradient, b.dataType, tA = true, tB = false, sA = aIsSparse, sB = gradIsSparse))
        case (false, true) =>
          Seq[OutputLike](
            helper(outputGradient, b, a.dataType, tA = false, tB = false, sA = gradIsSparse, sB = bIsSparse),
            helper(outputGradient, a, b.dataType, tA = true, tB = false, sA = gradIsSparse, sB = aIsSparse))
        case (true, false) =>
          Seq[OutputLike](
            helper(b, outputGradient, a.dataType, tA = false, tB = true, sA = bIsSparse, sB = gradIsSparse),
            helper(a, outputGradient, b.dataType, tA = false, tB = false, sA = aIsSparse, sB = gradIsSparse))
        case (true, true) =>
          Seq[OutputLike](
            helper(b, outputGradient, a.dataType, tA = true, tB = true, sA = bIsSparse, sB = gradIsSparse),
            helper(outputGradient, a, b.dataType, tA = true, tB = true, sA = gradIsSparse, sB = aIsSparse))
      }
    }

    private[this] def crossGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val u = op.inputs(0)
      val v = op.inputs(1)
      val outputGradient = outputGradients.head.toOutput
      Seq(cross(v, outputGradient), cross(outputGradient, u))
    }

    private[this] def complexGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val x = op.inputs(0)
      val y = op.inputs(1)
      val xShape = Basic.shape(x)
      val yShape = Basic.shape(y)
      val outputGradient = outputGradients.head.toOutput
      val (rx, ry) = Basic.broadcastGradientArguments(xShape, yShape)
      Seq(
        Basic.reshape(sum(real(outputGradient), rx), xShape),
        Basic.reshape(sum(imag(outputGradient), ry), yShape))
    }

    private[this] def realGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      Seq(complex(outputGradient, Basic.constant(0, outputGradient.dataType)))
    }

    private[this] def imagGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      val outputGradient = outputGradients.head.toOutput
      Seq(complex(Basic.constant(0, outputGradient.dataType), outputGradient))
    }

    private[this] def conjGradient(op: Op, outputGradients: Seq[OutputLike]): Seq[OutputLike] = {
      Seq(conjugate(outputGradients.head))
    }
  }

  /** Helper function for reduction ops that computes the reduction output shape, assuming `keepDims` is `true`.
    *
    * For example:
    * {{{
    *   // inputShape == [2, 3, 5, 7]
    *   // axes = [1, 2]
    *   reducedShape(inputShape, axes) ==> [2, 1, 1, 7]
    * }}}
    *
    * @param  inputShape Shape of the tensor being reduced.
    * @param  axes       Reduction axes.
    * @return One-dimensional tensor representing the reduction output shape, assuming `keepDims` is `true`.
    */
  private[api] def reducedShape(inputShape: Output, axes: Output): Output = {
    // Cast needed for SparseOutput reductions.
    val intInputShape = Cast.cast(inputShape, INT32)
    val inputRank = Basic.size(intInputShape, INT32)
    val reshapedAxes = {
      if (axes.rank == 0)
        Basic.reshape(axes, Tensor(1))
      else
        axes
    }
    val intAxes = floorMod(add(Cast.cast(reshapedAxes, INT32), inputRank), inputRank)
    val axesShape = Basic.shape(intAxes)
    DataFlow.dynamicStitch(
      Seq(range(Basic.constant(0), inputRank), intAxes),
      Seq(intInputShape, Basic.fill(shape = axesShape)(1)))
  }

  /** @define OpDocMathSelect
    *   The `select` op selects elements from `x` or `y`, depending on `condition`.
    *
    *   The `x`, and `y` tensors must have the same shape. The output tensor will also have the same shape.
    *
    *   The `condition` tensor must be a scalar if `x` and `y` are scalars. If `x` and `y` are vectors or higher rank,
    *   then `condition` must be either a scalar, or a vector with size matching the first dimension of `x`, or it must
    *   have the same shape as `x`.
    *
    *   The `condition` tensor acts as a mask that chooses, based on the value at each element, whether the
    *   corresponding element / row in the output should be taken from `x` (if true) or `y` (if false).
    *
    *   If `condition` is a vector and `x` and `y` are higher rank matrices, then it chooses which row (outer dimension)
    *   to copy from `x` and `y`. If `condition` has the same shape as `x` and `y`, then it chooses which element to
    *   copy from `x` and `y`.
    *
    *   For example:
    *   {{{
    *     // 'condition' tensor is [[true,  false], [false, true]]
    *     // 'x' is [[1, 2], [3, 4]]
    *     // 'y' is [[5, 6], [7, 8]]
    *     select(condition, x, y) ==> [[1, 6], [7, 4]]
    *
    *     // 'condition' tensor is [true, false]
    *     // 'x' is [[1, 2], [3, 4]]
    *     // 'y' is [[5, 6], [7, 8]]
    *     select(condition, x, y) ==> [[1, 2], [7, 8]]
    *   }}}
    * 
    * @define OpDocMathRange
    *   The `range` op constructs a sequence of numbers.
    *
    *   The op creates a sequence of numbers that begins at `start` and extends by increments of `delta` up to but not
    *   including `limit`. The data type of the resulting tensor is inferred from the inputs unless it is provided
    *   explicitly.
    *
    *   For example:
    *   {{{
    *     // 'start' is 3
    *     // 'limit' is 18
    *     // 'delta' is 3
    *     range(start, limit, delta) ==> [3, 6, 9, 12, 15]
    *
    *     // 'start' is 3
    *     // 'limit' is 1
    *     // 'delta' is -0.5
    *     range(start, limit, delta) ==> [3.0, 2.5, 2.0, 1.5]
    *   }}}
    *   
    * @define OpDocMathLinspace
    *   The `linspace` op generates values in an interval.
    *
    *   The op generates a sequence of `numberOfValues` evenly-spaced values beginning at `start`. If
    *   `numberOfValues > 1`, the values in the sequence increase by `(stop - start) / (numberOfValues - 1)`, so that the
    *   last value is exactly equal to `stop`.
    *
    *   For example:
    *   {{{
    *     linspace(10.0, 12.0, 3) ==> [10.0  11.0  12.0]
    *   }}
    *
    * @define OpDocMathAddN
    *   The `addN` op adds all input tensors element-wise.
    *
    * @define OpDocMathAccumulateN
    *   The `accumulateN` op adds all input tensors element-wise.
    *
    *   This op performs the same operation as the `addN` op, but it does not wait for all of its inputs to be ready 
    *   before beginning to sum. This can save memory if the inputs become available at different times, since the 
    *   minimum temporary storage is proportional to the output size, rather than the inputs size.
    *
    * @define OpDocMathAbs
    *   The `abs` op computes the absolute value of a tensor.
    *
    *   Given a tensor `x` of real numbers, the op returns a tensor containing the absolute value of each element in
    *   `x`. For example, if `x` is an input element and `y` is an output element, the op computes `y = |x|`.
    *
    *   Given a tensor `x` of complex numbers, the op returns a tensor of type `FLOAT32` or `FLOAT64` that is the
    *   magnitude value of each element in `x`. All elements in `x` must be complex numbers of the form `a + bj`. The
    *   magnitude is computed as `\sqrt{a^2 + b^2}`. For example:
    *   {{{
    *     // Tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
    *     abs(x) ==> [5.25594902, 6.60492229]
    *   }}}
    *
    * @define OpDocMathNegate
    *   The `negate` op computes the numerical negative value of a tensor element-wise. I.e., `y = -x`.
    *
    * @define OpDocMathReciprocal
    *   The `reciprocal` op computes the reciprocal value of a tensor element-wise. I.e., `y = 1 / x`.
    *
    * @define OpDocMathSquare
    *   The `square` op computes the square of a tensor element-wise. I.e., `y = x * x = x^2`.
    *
    * @define OpDocMathSqrt
    *   The `sqrt` op computes the square root of a tensor element-wise. I.e., `y = \sqrt{x} = x^{1/2}`.
    *
    * @define OpDocMathRsqrt
    *   The `rsqrt` op computes the reciprocal of the square root of a tensor element-wise. I.e.,
    *   `y = 1 / \sqrt{x} = 1 / x^{1/2}`.
    *
    * @define OpDocMathExp
    *   The `exp` op computes the exponential of a tensor element-wise. I.e., `y = \exp{x} = e^x`.
    *
    * @define OpDocMathExpm1
    *   The `expm1` op computes the exponential of a tensor minus `1` element-wise. I.e., `y = \exp{x} - 1`.
    *
    * @define OpDocMathLog
    *   The `log` op computes the logarithm of a tensor element-wise. I.e., `y = \log{x}`.
    *
    * @define OpDocMathLog1p
    *   The `log1p` op computes the logarithm of a tensor plus `1` element-wise. I.e., `y = \log{1 + x}`.
    *
    * @define OpDocMathSin
    *   The `sin` op computes the sine of a tensor element-wise. I.e., `y = \sin{x}`.
    *
    * @define OpDocMathCos
    *   The `cos` op computes the cosine of a tensor element-wise. I.e., `y = \cos{x}`.
    *
    * @define OpDocMathTan
    *   The `tan` op computes the tangent of a tensor element-wise. I.e., `y = \tan{x}`.
    *
    * @define OpDocMathAsin
    *   The `asin` op computes the inverse sine of a tensor element-wise. I.e., `y = \asin{x}`.
    *
    * @define OpDocMathAcos
    *   The `acos` op computes the inverse cosine of a tensor element-wise. I.e., `y = \acos{x}`.
    *
    * @define OpDocMathAtan
    *   The `atan` op computes the inverse tangent of a tensor element-wise. I.e., `y = \atan{x}`.
    *
    * @define OpDocMathSinh
    *   The `sinh` op computes the hyperbolic sine of a tensor element-wise. I.e., `y = \sinh{x}`.
    *
    * @define OpDocMathCosh
    *   The `cosh` op computes the hyperbolic cosine of a tensor element-wise. I.e., `y = \cosh{x}`.
    *
    * @define OpDocMathTanh
    *   The `tanh` op computes the hyperbolic tangent of a tensor element-wise. I.e., `y = \tanh{x}`.
    *
    * @define OpDocMathAsinh
    *   The `asinh` op computes the inverse hyperbolic sine of a tensor element-wise. I.e., `y = \asinh{x}`.
    *
    * @define OpDocMathAcosh
    *   The `acosh` op computes the inverse hyperbolic cosine of a tensor element-wise. I.e., `y = \acosh{x}`.
    *
    * @define OpDocMathAtanh
    *   The `atanh` op computes the inverse hyperbolic tangent of a tensor element-wise. I.e., `y = \atanh{x}`.
    *
    * @define OpDocMathLogGamma
    *   The `logGamma` op computes the logarithm of the absolute value of the Gamma function applied element-wise on a
    * 	tensor. I.e., `y = \log{|\Gamma{x}|}`.
    *
    * @define OpDocMathDigamma
    *   The `digamma` op computes the derivative of the logarithm of the absolute value of the Gamma function applied
    * 	element-wise on a tensor (i.e., the digamma or Psi function). I.e., `y = \partial\log{|\Gamma{x}|}`.
    *
    * @define OpDocMathErf
    *   The `erf` op computes the Gaussian error function element-wise on a tensor.
    *
    * @define OpDocMathErfc
    *   The `erfc` op computes the complementary Gaussian error function element-wise on a tensor.
    *
    * @define OpDocMathSigmoid
    *   The `sigmoid` op computes the sigmoid function element-wise on a tensor.
    *
    *   Specifically, `y = 1 / (1 + exp(-x))`.
    *
    * @define OpDocMathLogSigmoid
    *   The `logSigmoid` op computes the log-sigmoid function element-wise on a tensor.
    *
    *   Specifically, `y = log(1 / (1 + exp(-x)))`.  For numerical stability, we use `y = -tf.nn.softplus(-x)`.
    *
    * @define OpDocMathSign
    *   The `sign` op computes an element-wise indication of the sign of a tensor.
    *		
    * 	I.e., `y = sign(x) = -1` if `x < 0`; `0` if `x == 0`; `1` if `x > 0`.
    *	
    * 	Zero is returned for `NaN` inputs.
    *		
    * 	For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.
    *	
    * @define OpDocMathRound
    *   The `round` op computes the round value of a tensor element-wise.
    *
    * 	Rounds half to even. Also known as bankers rounding. If you want to round according to the current system 
    *		rounding mode use the [[roundInt]] op instead.
    *
    * 	For example:
    * 	{{{
    *   	// 'a' is [0.9, 2.5, 2.3, 1.5, -4.5]
    *   	round(a) ==> [1.0, 2.0, 2.0, 2.0, -4.0]
    * 	}}}
    *	
    * @define OpDocMathRoundInt
    *   The `roundInt` op computes the round value of a tensor element-wise.
    *		
    * 	If the result is midway between two representable values, the even representable is chosen.
    *
    * 	For example:
    * 	{{{
    *   	roundInt(-1.5) ==> -2.0
    *   	roundInt(0.5000001) ==> 1.0
    *   	roundInt([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
    * 	}}}
    *	
    * @define OpDocMathFloor
    *   The `floor` op computes the largest integer not greater than the current value of a tensor, element-wise.
    *	
    * @define OpDocMathCeil
    *   The `ceil` op computes the smallest integer not greater than the current value of a tensor, element-wise.
    *	
    * @define OpDocMathIsNaN
    *   The `isNaN` op returns a boolean tensor indicating which elements of a tensor are NaN-valued.
    *	
    * @define OpDocMathIsInf
    *   The `isInf` op returns a boolean tensor indicating which elements of a tensor are Inf-valued.
    *	
    * @define OpDocMathIsFinite
    *   The `isFinite` op returns a boolean tensor indicating which elements of a tensor are finite-valued.
		* 
		* @define OpDocMathAdd
		*   The `add` op adds two tensors element-wise. I.e., `z = x + y`.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathSubtract
		*   The `subtract` op subtracts two tensors element-wise. I.e., `z = x - y`.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathMultiply
		*   The `multiply` op multiplies two tensors element-wise. I.e., `z = x * y`.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathDivide
		*   The `divide` op divides two tensors element-wise. I.e., `z = x / y`.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathFloorDivide
		*   The `floorDivide` op floor-divides two tensors element-wise. I.e., `z = x // y`.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathTruncateDivide
		*   The `truncateDivide` op truncate-divides two tensors element-wise.
		*   
		*   Truncation designates that negative numbers will round fractional quantities toward zero. I.e. `-7 / 5 = 1`. 
		*   This matches C semantics but it is different than Python semantics. See `floorDivide` for a division function 
		*   that matches Python semantics.
    *
    *   I.e., `z = x / y`, for `x` and `y` being integer tensors.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathRealDivide
		*   The `realDivide` op divides two real tensors element-wise.
		*   
		*   If `x` and `y` are real-valued tensors, the op will return the floating-point division.
    *
    *   I.e., `z = x / y`, for `x` and `y` being real tensors.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathSquaredDifference
		*   The `squaredDifference` op computes the squared difference between two tensors element-wise. 
		*   I.e., `z = (x - y) * (x - y)`.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathMod
		*   The `mod` op computes the remainder of the division between two tensors element-wise.
    *
    *   The op emulates C semantics in that the result is consistent with a truncating divide.
    *   E.g., `truncate(x / y) * y + truncateMod(x, y) = x`.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathFloorMod
		*   The `floorMod` op computes the remainder of the division between two tensors element-wise.
    *
    *   When `x < 0` xor `y < 0` is true, the op follows Python semantics in that the result here is 
    *   consistent with a flooring divide. E.g., `floor(x / y) * y + mod(x, y) = x`.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathTruncateMod
		*   The `truncateMod` op computes the remainder of the division between two tensors element-wise.
    *
    *   The op emulates C semantics in that the result here is consistent with a truncating divide.
    *   E.g., `truncate(x / y) * y + truncateMod(x, y) = x`.
    *   
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathPow
		*   The `pow` op computes the power of one tensor raised to another, element-wise.
    *
    *   Given a tensor `x` and a tensor `y`, the op computes `x^y` for the corresponding elements in `x` 
    *   and `y`.
    *
    *   For example:
    *   {{{
    *     // Tensor 'x' is [[2, 2], [3, 3]]
    *     // Tensor 'y' is [[8, 16], [2, 3]]
    *     pow(x, y) ==> [[256, 65536], [9, 27]]
    *   }}}
		* 
		* @define OpDocMathIgammac
		*   The `igammac` op computes the upper regularized incomplete Gamma function `Q(a, x)`.
    *
    *   The upper regularized incomplete Gamma function is defined as:
    *
    *   `Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)`, where:
    *
    *   `Gamma(a, x) = \int_{x}^{\infty} t^{a-1} exp(-t) dt`
    *
    *   is the upper incomplete Gama function.
    *
    *   Note that, above, `P(a, x)` (`Igamma`) is the lower regularized complete Gamma function.
		* 
		* @define OpDocMathIgamma
		*   The `igamma` op computes the lower regularized incomplete Gamma function `Q(a, x)`.
    *
    *   The lower regularized incomplete Gamma function is defined as:
    *
    *   `P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)`, where:
    *
    *   `Gamma(a, x) = \int_{0}^{x} t^{a-1} exp(-t) dt`
    *
    *   is the lower incomplete Gamma function.
    *
    *   Note that, above, `Q(a, x)` (`Igammac`) is the upper regularized complete Gamma function.
		* 
		* @define OpDocMathZeta
		*   The `zeta` op computes the Hurwitz zeta function `\zeta(x, q)`.
    *
    *   The Hurwitz zeta function is defined as:
    *
    *   `\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}`.
		* 
		* @define OpDocMathPolygamma
		*   The `polygamma` op computes the polygamma function `\psi^{(n)}(x)`.
    *
    *   The polygamma function is defined as:
    *
    *   `\psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)`, where `\psi(x)` is the digamma function.
		* 
		* @define OpDocMathAtan2
		*   The `atan2` op computes the inverse tangent of `x / y` element-wise, respecting signs of the arguments.
    *
    *   The op computes the angle `\theta \in [-\pi, \pi]` such that `y = r \cos(\theta)` and 
    *   `x = r \sin(\theta)`, where `r = \sqrt(x^2 + y^2)`.
    *
    * @define OpDocMathMinimum
    *   The `minimum` op returns the element-wise minimum between two tensors. I.e., `z = x < y ? x : y`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
		* 
		* @define OpDocMathMaximum
		*   The `maximum` op returns the element-wise maximum between two tensors. I.e., `z = x > y ? x : y`.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathIncompleteBeta
    *   The `incompleteBeta` op computes the regularized incomplete beta integral `I_x(a, b)`.
    *
    *   The regularized incomplete beta integral is defined as:
    *
    *   `I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}`, where:
    *
    *   `B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt`
    *
    *   is the incomplete beta function and `B(a, b)` is the *complete* beta function.
    * 
    * @define OpDocMathLogicalNot
    *   The `logicalNot` op computes the truth value of `!x` element-wise.
    * 
    * @define OpDocMathLogicalAnd
    *   The `logicalAnd` op computes the truth value of `x && y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathLogicalOr
    *   The `logicalOr` op computes the truth value of `x || y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathLogicalXOr
    *   The `logicalXOr` op computes the truth value of `(x || y) && !(x && y)` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathEqual
    *   The `equal` op computes the truth value of `x == y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathNotEqual
    *   The `notEqual` op computes the truth value of `x != y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathApproximatelyEqual
    *   The `approximatelyEqual` op computes the truth value of `abs(x - y) < tolerance` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathLess
    *   The `less` op computes the truth value of `x < y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathLessEqual
    *   The `lessEqual` op computes the truth value of `x <= y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathGreater
    *   The `greater` op computes the truth value of `x > y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathGreaterEqual
    *   The `greaterEqual` op computes the truth value of `x >= y` element-wise.
    *
    *   NOTE: This op supports broadcasting. More information about broadcasting can be found
    *   [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
    * 
    * @define OpDocMathSum
    *   The `sum` op computes the sum of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced 
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1, 1, 1]], [1, 1, 1]]
    *     sum(x) ==> 6
    *     sum(x, 0) ==> [2, 2, 2]
    *     sum(x, 1) ==> [3, 3]
    *     sum(x, 1, keepDims = true) ==> [[3], [3]]
    *     sum(x, [0, 1]) ==> 6
    *   }}}
    * 
    * @define OpDocMathMean
    *   The `mean` op computes the mean of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced 
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *     mean(x) ==> 1.5
    *     mean(x, 0) ==> [1.5, 1.5]
    *     mean(x, 1) ==> [1.0, 2.0]
    *   }}}
    * 
    * @define OpDocMathProd
    *   The `prod` op computes the product of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced 
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1, 1, 1]], [1, 1, 1]]
    *     prod(x) ==> 1
    *     prod(x, 0) ==> [1, 1, 1]
    *     prod(x, 1) ==> [1, 1]
    *     prod(x, 1, keepDims = true) ==> [[1], [1]]
    *     prod(x, [0, 1]) ==> 1
    *   }}}
    * 
    * @define OpDocMathMin
    *   The `min` op computes the minimum of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced 
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *     min(x) ==> 1.0
    *     min(x, 0) ==> [1.0, 1.0]
    *     min(x, 1) ==> [1.0, 2.0]
    *   }}}
    * 
    * @define OpDocMathMax
    *   The `max` op computes the maximum of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced 
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1.0, 1.0], [2.0, 2.0]]
    *     max(x) ==> 2.0
    *     max(x, 0) ==> [2.0, 2.0]
    *     max(x, 1) ==> [1.0, 2.0]
    *   }}}
    * 
    * @define OpDocMathAll
    *   The `all` op computes the logical AND of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced 
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[true, true], [false, false]]
    *     all(x) ==> false
    *     all(x, 0) ==> [false, false]
    *     all(x, 1) ==> [true, false]
    *   }}}
    * 
    * @define OpDocMathAny
    *   The `any` op computes the logical OR of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced 
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[true, true], [false, false]]
    *     any(x) ==> true
    *     any(x, 0) ==> [true, true]
    *     any(x, 1) ==> [true, false]
    *   }}}
    * 
    * @define OpDocMathLogSumExp
    *   The `logSumExp` op computes the log-sum-exp of elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced 
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[0, 0, 0], [0, 0, 0]]
    *     logSumExp(x) ==> log(6)
    *     logSumExp(x, 0) ==> [log(2), log(2), log(2)]
    *     logSumExp(x, 1) ==> [log(3), log(3)]
    *     logSumExp(x, 1, keepDims = true) ==> [[log(3)], [log(3)]]
    *     logSumExp(x, [0, 1]) ==> log(6)
    *   }}}
    * 
    * @define OpDocMathCountNonZero
    *   The `countNonZero` op computes the number of non-zero elements across axes of a tensor.
    *
    *   Reduces `input` along the axes given in `axes`. Unless `keepDims` is `true`, the rank of the tensor is reduced 
    *   by 1 for each entry in `axes`. If `keepDims` is `true`, the reduced axes are retained with size 1.
    *
    *   If `axes` is `null`, then all axes are reduced, and a tensor with a single element is returned.
    *   
		*   '''IMPORTANT NOTE:''' Floating point comparison to zero is done by exact floating point equality check. Small
    *   values are '''not''' rounded to zero for the purposes of the non-zero check.
		*   
    *   For example:
    *   {{{
    *     // 'x' is [[0, 1, 0], [1, 1, 0]]
    *     countNonZero(x) ==> 3
    *     countNonZero(x, 0) ==> [1, 2, 0]
    *     countNonZero(x, 1) ==> [1, 2]
    *     countNonZero(x, 1, keepDims = true) ==> [[1], [2]]
    *     countNonZero(x, [0, 1]) ==> 3
    *   }}}
    *
    *   '''IMPORTANT NOTE:''' Strings are compared against zero-length empty string `""`. Any string with a size greater
    *   than zero is already considered as nonzero.
    *
    *   For example:
    *   {{{
    *     // 'x' is ["", "a", "  ", "b", ""]
    *     countNonZero(x) ==> 3 // "a", "  ", and "b" are treated as nonzero strings.
    *   }}}
    * 
    * @define OpDocMathArgmax
    *   The `argmax` op returns the indices with the largest value across axes of a tensor.
    *
    *   Note that in case of ties the identity of the return value is not guaranteed.
    * 
    * @define OpDocMathArgmin
    *   The `argmin` op returns the indices with the smallest value across axes of a tensor.
    *
    *   Note that in case of ties the identity of the return value is not guaranteed.
    * 
    * @define OpDocMathBinCount
    *   The `binCount` op counts the number of occurrences of each value in an integer tensor.
    *
    *   If `minLength` and `maxLength` are not provided, the op returns a vector with length `max(input) + 1`, if 
    *   `input` is non-empty, and length `0` otherwise.
    *
    *   If `weights` is not `null`, then index `i` of the output stores the sum of the value in `weights` at each 
    *   index where the corresponding value in `input` is equal to `i`.
    * 
    * @define OpDocMathCumsum
    *   The `cumsum` op computes the cumulative sum of the tensor along an axis.
    *
    *   By default, the op performs an inclusive cumulative sum, which means that the first element of the input is
    *   identical to the first element of the output:
    *   {{{
    *     cumsum([a, b, c]) ==> [a, a + b, a + b + c]
    *   }}}
    *
    *   By setting the `exclusive` argument to `true`, an exclusive cumulative sum is performed instead:
    *   {{{
    *     cumsum([a, b, c], exclusive = true) ==> [0, a, a + b]
    *   }}}
    *
    *   By setting the `reverse` argument to `true`, the cumulative sum is performed in the opposite direction:
    *   {{{
    *     cumsum([a, b, c], reverse = true) ==> [a + b + c, b + c, c]
    *   }}}
    *
    *   This is more efficient than using separate [[Basic.reverse]] ops.
    *
    *   The `reverse` and `exclusive` arguments can also be combined:
    *   {{{
    *     cumsum([a, b, c], exclusive = true, reverse = true) ==> [b + c, c, 0]
    *   }}}
    * 
    * @define OpDocMathCumprod
    *   The `cumprod` op computes the cumulative product of the tensor along an axis.
    *
    *   By default, the op performs an inclusive cumulative product, which means that the first element of the input 
    *   is identical to the first element of the output:
    *   {{{
    *    cumprod([a, b, c]) ==> [a, a * b, a * b * c]
    *   }}} 
    *
    *   By setting the `exclusive` argument to `true`, an exclusive cumulative product is performed instead:
    *   {{{
    *     cumprod([a, b, c], exclusive = true) ==> [0, a, a * b]
    *   }}}
    *
    *   By setting the `reverse` argument to `true`, the cumulative product is performed in the opposite direction:
    *   {{{
    *     cumprod([a, b, c], reverse = true) ==> [a * b * c, b * c, c]
    *   }}}
    *
    *   This is more efficient than using separate [[Basic.reverse]] ops.
    *
    *   The `reverse` and `exclusive` arguments can also be combined:
    *   {{{
    *     cumprod([a, b, c], exclusive = true, reverse = true) ==> [b * c, c, 0]
    *   }}}
    * 
    * @define OpDocMathSegmentSum
    *   The `segmentSum` op computes the sum along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \sum_{j...} data(j,...)` where the sum is over all `j` such 
    *   that `segmentIndices(j) == i`. Unlike `unsortedSegmentSum`, `segmentIndices` need be sorted.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathSegmentMean
    *   The `segmentMean` op computes the mean along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \frac{sum_{j...} data(j,...)}{N}` where the sum is over 
    *   all `j` such that `segmentIndices(j) == i` and `N` is the total number of values being summed. Unlike
    *   `unsortedSegmentMean`, `segmentIndices` need to be sorted.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathSegmentProd
    *   The `segmentProd` op computes the product along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \prod_{j...} data(j,...)` where the product is over all `j` 
    *   such that `segmentIndices(j) == i`. Unlike `unsortedSegmentProd`, `segmentIndices` need be sorted.
    *
    *   If the product if empty for a given segment index `i`, `output(i)` is set to `1`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathSegmentMin
    *   The `segmentMin` op computes the min along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \min_{j...} data(j,...)` where the min is over all `j` 
    *   such that `segmentIndices(j) == i`. Unlike `unsortedSegmentMin`, `segmentIndices` need be sorted.
    *
    *   If the min if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathSegmentMax
    *   The `segmentMax` op computes the max along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \max_{j...} data(j,...)` where the max is over all `j` 
    *   such that `segmentIndices(j) == i`. Unlike `unsortedSegmentMax`, `segmentIndices` need be sorted.
    *
    *   If the max if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentSum
    *   The `unsortedSegmentSum` op computes the sum along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \sum_{j...} data(j...)` where the sum is over all `j`
    *   such that `segmentIndices(j) == i`. Unlike `segmentSum`, `segmentIndices` need not be sorted and need not
    *   cover all values in the full range of valid values.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentMean
    *   The `unsortedSegmentMean` op computes the mean along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \frac{\sum_{j...} data(j...)}{N}` where the sum is over 
    *   all `j` such that `segmentIndices(j) == i` and `N` is the total number of values being summed. Unlike 
    *   `segmentSum`, `segmentIndices` need not be sorted and need not cover all values in the full range of valid 
    *   values.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentProd
    *   The `unsortedSegmentProd` op computes the product along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \prod_{j...} data(j...)` where the product is over all `j`
    *   such that `segmentIndices(j) == i`. Unlike `segmentProd`, `segmentIndices` need not be sorted and need not
    *   cover all values in the full range of valid values.
    *
    *   If the product if empty for a given segment index `i`, `output(i)` is set to `1`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathUnsortedSegmentMin
    *   The `unsortedSegmentMin` op computes the min along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \min_{j...} data(j...)` where the min is over all `j` 
    *   such that `segmentIndices(j) == i`. Unlike `segmentMin`, `segmentIndices` need not be sorted and need not 
    *   cover all values in the full range of valid values.
    *
    *   If the min if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathUnsortedSegmentMax
    *   The `unsortedSegmentMax` op computes the max along segments of a tensor.
    *
    *   The op computes a tensor such that `output(i) = \max_{j...} data(j...)` where the max is over all `j` 
    *   such that `segmentIndices(j) == i`. Unlike `segmentMax`, `segmentIndices` need not be sorted and need not 
    *   cover all values in the full range of valid values.
    *
    *   If the max if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    *
    * @define OpDocMathUnsortedSegmentSqrtN
    *   The `unsortedSegmentSqrtN` op computes the sum along segments of a tensor, divided by the square root of 
    *   number of elements being summed.
    *
    *   The op computes a tensor such that `output(i) = \frac{\sum_{j...} data(j...)}{\sqrt{N}}` where the sum is 
    *   over all `j` such that `segmentIndices(j) == i` and `N` is the total number of values being summed.
    *
    *   If the sum if empty for a given segment index `i`, `output(i)` is set to `0`.
    *
    *   `segmentsNumber` should equal the number of distinct segment indices.
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathSparseSegmentSum
    *   The `sparseSegmentSum` op computes the sum along sparse segments of a tensor.
    *
    *   The op is similar to that of [[segmentSum]], with the difference that `segmentIndices` can have rank less 
    *   than `data`'s first dimension, selecting a subset of dimension `0`, specified by `indices`. `segmentIndices` is
    *   allowed to have missing indices, in which case the output will be zeros at those indices. In those cases,
    *   `numSegments` is used to determine the size of the output.
    *
    *   For example:
    *   {{{
    *     // 'c' is [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]]
    *
    *     // Select two rows, one segment.
    *     sparseSegmentSum(c, Tensor(0, 1), Tensor(0, 0)) ==> [[0, 0, 0, 0]]
    *
    *     // Select two rows, two segments.
    *     sparseSegmentSum(c, Tensor(0, 1), Tensor(0, 1)) ==> [[1, 2, 3, 4], [-1, -2, -3, -4]]
    *
    *     // Select all rows, two segments.
    *     sparseSegmentSum(c, Tensor(0, 1, 2), Tensor(0, 0, 1)) ==> [[0, 0, 0, 0], [5, 6, 7, 8]]
    *     // which is equivalent to:
    *     segmentSum(c, Tensor(0, 0, 1))
    *   }}}
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathSparseSegmentMean
    *   The `sparseSegmentMean` op computes the mean along sparse segments of a tensor.
    *
    *   The op is similar to that of [[segmentMean]], with the difference that `segmentIndices` can have rank less 
    *   than `data`'s first dimension, selecting a subset of dimension `0`, specified by `indices`. `segmentIndices` is
    *   allowed to have missing indices, in which case the output will be zeros at those indices. In those cases,
    *   `numSegments` is used to determine the size of the output.
    *
    *   For example:
    *   {{{
    *     // 'c' is [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]]
    *
    *     // Select two rows, one segment.
    *     sparseSegmentMean(c, Tensor(0, 1), Tensor(0, 0)) ==> [[0, 0, 0, 0]]
    *
    *     // Select two rows, two segments.
    *     sparseSegmentMean(c, Tensor(0, 1), Tensor(0, 1)) ==> [[1, 2, 3, 4], [-1, -2, -3, -4]]
    *
    *     // Select all rows, two segments.
    *     sparseSegmentMean(c, Tensor(0, 1, 2), Tensor(0, 0, 1)) ==> [[0, 0, 0, 0], [5, 6, 7, 8]]
    *     // which is equivalent to:
    *     segmentMean(c, Tensor(0, 0, 1))
    *   }}}
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathSparseSegmentSumSqrtN
    *   The `sparseSegmentSumSqrtN` op computes the sum along sparse segments of a tensor, divided by the square 
    *   root of the number of elements being summed. `segmentIndices` is allowed to have missing indices, in which case
    *   the output will be zeros at those indices. In those cases, `numSegments` is used to determine the size of the
    *   output.
    *
    *   Similar to [[sparseSegmentSum]].
    *
    *   The result tensor has the same data type as `data`, but its first dimension size is equal to the number of
    *   distinct segment indices.
    * 
    * @define OpDocMathDiag
    *   The `diag` op constructs a diagonal tensor using the provided diagonal values.
    *
    *   Given a `diagonal`, the op returns a tensor with that `diagonal` and everything else padded with zeros. The
    *   diagonal is computed as follows:
    *
    *   Assume that `diagonal` has shape `[D1,..., DK]`. Then the output tensor, `output`, is a rank-`2K` tensor with
    *   shape `[D1, ..., DK, D1, ..., DK]`, where `output(i1, ..., iK, i1, ..., iK) = diagonal(i1, ..., iK)` and `0`
    *   everywhere else.
    *
    *   For example:
    *   {{{
    *     // 'diagonal' is [1, 2, 3, 4]
    *     diag(diagonal) ==> [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
    *   }}}
    *
    *   This op is the inverse of [[diagPart]].
    * 
    * @define OpDocMathDiagPart
    *   The `diagPart` op returns the diagonal part of a tensor.
    *
    *   The op returns a tensor with the `diagonal` part of the `input`. The `diagonal` part is computed as follows:
    *
    *   Assume `input` has shape `[D1, ..., DK, D1, ..., DK]`. Then the output is a rank-`K` tensor with shape
    *   `[D1,..., DK]`, where `diagonal(i1, ..., iK) = output(i1, ..., iK, i1, ..., iK)`.
    *
    *   For example:
    *   {{{
    *     // 'input' is [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
    *     diagPart(input) ==> [1, 2, 3, 4]
    *   }}}
    *
    *   This op is the inverse of [[diag]].
    * 
    * @define OpDocMathMatrixDiag
    *   The `matrixDiag` op returns a batched diagonal tensor with the provided batched diagonal values.
    *
    *   Given a `diagonal`, the op returns a tensor with that `diagonal` and everything else padded with zeros. Assuming
    *   that `diagonal` has `k` dimensions `[I, J, K, ..., N]`, the output is a tensor of rank `k + 1` with dimensions
    *   `[I, J, K, ..., N, N]`, where: `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.
    *
    *   For example:
    *   {{{
    *     // 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]] (shape = [2, 4])
    *     matrixDiag(diagonal) ==> [[[1, 0, 0, 0]
    *                                [0, 2, 0, 0]
    *                                [0, 0, 3, 0]
    *                                [0, 0, 0, 4]],
    *                               [[5, 0, 0, 0]
    *                                [0, 6, 0, 0]
    *                                [0, 0, 7, 0]
    *                                [0, 0, 0, 8]]]  // with shape [2, 4, 4]
    *   }}}
    * 
    * @define OpDocMathMatrixSetDiag
    *   The `matrixSetDiag` op returns a batched matrix tensor with new batched diagonal values.
    *
    *   Given `input` and `diagonal`, the op returns a tensor with the same shape and values as `input`, except for the
    *   main diagonal of its innermost matrices. These diagonals will be overwritten by the values in `diagonal`. 
    *   Assuming that `input` has `k + 1` dimensions, `[I, J, K, ..., M, N]`, and `diagonal` has `k` dimensions,
    *   `[I, J, K, ..., min(M, N)]`, then the output is a tensor of rank `k + 1` with dimensions `[I, J, K, ..., M, N]`,
    *   where:
    *     - `output[i, j, k, ..., m, n] == diagonal[i, j, k, ..., n]`, for `m == n`, and
    *     - `output[i, j, k, ..., m, n] == input[i, j, k, ..., m, n]`, for `m != n`.
    * 
    * @define OpDocMathMatrixDiagPart
    *   The `matrixDiagPart` op returns the batched diagonal part of a batched tensor.
    *
    *   The op returns a tensor with the `diagonal` part of the batched `input`. Assuming that `input` has `k` 
    *   dimensions, `[I, J, K, ..., M, N]`, then the output is a tensor of rank `k - 1` with dimensions 
    *   `[I, J, K, ..., min(M, N)]`, where `diagonal[i, j, k, ..., n] == input[i, j, k, ..., n, n]`.
    *
    *   Note that `input` must have rank of at least `2`.
    *
    *   For example:
    *   {{{
    *     // 'input' is:
    *     //   [[[1, 0, 0, 0]
    *     //     [0, 2, 0, 0]
    *     //     [0, 0, 3, 0]
    *     //     [0, 0, 0, 4]],
    *     //    [[5, 0, 0, 0]
    *     //     [0, 6, 0, 0]
    *     //     [0, 0, 7, 0]
    *     //     [0, 0, 0, 8]]]  with shape [2, 4, 4]
    *     matrixDiagPart(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]  // with shape [2, 4]
    *   }}}
    * 
    * @define OpDocMathMatrixBandPart
    *   The `matrixBandPart` op copies a tensor, while setting everything outside a central band in each innermost 
    *   matrix of the tensor, to zero.
    *   
    *   Assuming that `input` has `k` dimensions, `[I, J, K, ..., M, N]`, the output is a tensor with the same shape,
    *   where `band[i, j, k, ..., m, n] == indicatorBand(m, n) * input[i, j, k, ..., m, n]`. The indicator function is
    *   defined as:
    *   {{{
    *     indicatorBand(m, n) = (numSubDiagonals < 0 || m - n <= numSubDiagonals) &&
    *                           (numSuperDiagonals < 0 || n - m <= numSuperDiagonals)
    *   }}}
    *   
    *   For example:
    *   {{{
    *     // 'input' is:
    *     //   [[ 0,  1,  2, 3]
    *     //    [-1,  0,  1, 2]
    *     //    [-2, -1,  0, 1]
    *     //    [-3, -2, -1, 0]]
    *     matrixBandPart(input, 1, -1) ==> [[ 0,  1,  2, 3]
    *                                       [-1,  0,  1, 2]
    *                                       [ 0, -1,  0, 1]
    *                                       [ 0,  0, -1, 0]]
    *     matrixBandPart(input, 2, 1) ==>  [[ 0,  1,  0, 0]
    *                                       [-1,  0,  1, 0]
    *                                       [-2, -1,  0, 1]
    *                                       [ 0, -2, -1, 0]]
    *   }}}
    * 
    *   Useful special cases:
    *   {{{
    *     matrixBandPart(input, 0, -1) ==> Upper triangular part
    *     matrixBandPart(input, -1, 0) ==> Lower triangular part
    *     matrixBandPart(input, 0, 0)  ==> Diagonal
    *   }}}
    * 
    * @define OpDocMathTrace
    *   The `trace` op computes the trace of a tensor.
    *
    *   The trace of a tensor is defined as the sum along the main diagonal of each inner-most matrix in it. 
    *   If the tensor is of rank `k` with shape `[I, J, K, ..., L, M, N]`, then output is a tensor of rank 
    *   `k - 2` with dimensions `[I, J, K, ..., L]` where: 
    *   `output[i, j, k, ..., l] = trace(x[i, j, i, ..., l, :, :])`.
    *
    *   For example:
    *   {{{
    *     // 'x' is [[1, 2], [3, 4]]
    *     trace(x) ==> 5
    *
    *     // 'x' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    *     trace(x) ==> 15
    *
    *     // 'x' is [[[ 1,  2,  3],
    *     //          [ 4,  5,  6],
    *     //          [ 7,  8,  9]],
    *     //         [[-1, -2, -3],
    *     //          [-4, -5, -6],
    *     //          [-7, -8, -9]]]
    *     trace(x) ==> [15, -15]
    *   }}}
    * 
    * @define OpDocMathScalarMul
    *   The `scalarMul` op multiplies a scalar tensor with another, potentially sparse, tensor.
    *
    *   This function is intended for use in gradient code which might deal with [[OutputIndexedSlices]] objects, 
    *   which are easy to multiply by a scalar but more expensive to multiply with arbitrary tensors.
    * 
    * @define OpDocMathMatmul
    *   The `matmul` op multiples two matrices.
    *
    *   The inputs must, following any transpositions, be tensors of rank >= 2, where the inner 2 dimensions specify 
    *   valid matrix multiplication arguments and any further outer dimensions match.
    *   
    *   Note that this op corresponds to a matrix product and not an element-wise product. For example:
    *   `output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`, for all indices `i` and `j`.
    *
    *   Both matrices must be of the same data type. The supported types are: `BFLOAT16`, `FLOAT16`, `FLOAT32`, 
    *   `FLOAT64`, `INT32`, `COMPLEX64`, and `COMPLEX128`.
    *
    *   Either matrix can be transposed and/or conjugated on the fly by setting one of the corresponding flags to 
    *   `true`. These are set to `false` by default.
    *   
    *   If one or both of the matrices contain a lot of zeros, a more efficient multiplication algorithm can be used 
    *   by setting the corresponding `aIsSparse` or `bIsSparse` flag to `true`. These are also set to `false` by 
    *   default. This optimization is only available for plain matrices (i.e., rank-2 tensors) with data type 
    *   `BFLOAT16` or `FLOAT32`. The break-even for using this versus a dense matrix multiply on one platform was 
    *   30% zero values in the sparse matrix. The gradient computation of the sparse op will only take advantage of 
    *   sparsity in the input gradient when that gradient comes from a ReLU.
    *   
    *   For example:
    *   {{{
    *     // 2-D tensor 'a' is [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    *
    *     // 2-D tensor 'b' is [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    *
    *     matmul(a, b) ==> [[58.0, 64.0], [139.0, 154.0]]
    *
    *     // 3-D tensor 'a' is [[[ 1.0,  2.0,  3.0],
    *     //                     [ 4.0,  5.0,  6.0]],
    *     //                    [[ 7.0,  8.0,  9.0],
    *     //                     [10.0, 11.0, 12.0]]]
    *
    *     // 3-D tensor 'b' is [[[13.0, 14.0],
    *     //                     [15.0, 16.0],
    *     //                     [17.0, 18.0]],
    *     //                    [[19.0, 20.0],
    *     //                     [21.0, 22.0],
    *     //                     [23.0, 24.0]]]
    *
    *     matmul(a, b) ==> [[[ 94.0, 100.0], [229.0, 244.0]],
    *                       [[508.0, 532.0], [697.0, 730.0]]]
    *   }}}
    * 
    * @define OpDocMathCross
    *   The `cross` op computes the pairwise cross product between two tensors.
    *
    *   `a` and `b` must have the same shape; they can either be simple 3-element vectors, or have any shape 
    *   where the innermost dimension size is 3. In the latter case, each pair of corresponding 3-element vectors 
    *   is cross-multiplied independently.
    *
    * @define OpDocMathTensorDot
    *   The `tensorDot` op computes the tensor contraction of two tensors along the specified axes.
    *   
    *   A tensor contraction sums the product of elements from `a` and `b` over the indices specified by `axesA` and 
    *   `axesB`. The axis `axesA(i)` of `a` must have the same dimension as the axis `axesB(i)` of `b` for all `i` in 
    *   `[0, aAxes.size)`. The tensors/sequences (depending on whether the dynamic version of the op is being used) 
    *   `axesA` and `axesB` must have identical length and consist of unique integers that specify valid axes for each 
    *   of the tensors. This operation corresponds to `numpy.tensordot(a, b, axes)` in Python.
    *   
    *   If `numAxes` is provided instead of `axesA` and `axesB`, then the contraction is performed over the last 
    *   `numAxes` axes of `a` and the first `numAxes` axes of `b`, in order.
    *   
    *   Example 1: When `a` and `b` are matrices (rank 2), the case `numAxes = 1` is equivalent to matrix 
    *              multiplication.
    *   Example 2: When `a` and `b` are matrices (rank 2), the case `axesA = [1]` and `axesB = [0]` is equivalent to 
    *              matrix multiplication.
    *   Example 3: Suppose that `a_{ijk}` and `b_{lmn}` represent two tensors of rank 3. Then, the case `axesA = [0]` 
    *              and `axesB = [2]` results in the rank 4 tensor `c_{jklm}` whose entry corresponding to the indices 
    *              `(j, k, l, m)` is given by: `c_{jklm} = \sum_i a_{ijk} b_{lmi}`. In general, 
    *              `rank(result) = rank(a) + rank(b) - 2 * axesA.size`.
    * 
    * @define OpDocMathComplex
    *   The `complex` op converts two real tensors to a complex tensor.
    *
    *   Given a tensor `real` representing the real part of a complex number, and a tensor `imag` representing the
    *   imaginary part of a complex number, the op returns complex numbers element-wise of the form `a + bj`, where *a*
    *   represents the `real` part and *b* represents the `imag` part. The input tensors `real` and `imag` must have the
    *   same shape and data type.
    *
    *   For example:
    *   {{{
    *     // 'real' is [2.25, 3.25]
    *     // 'imag' is [4.75, 5.75]
    *     complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
    *   }}}
    * 
    * @define OpDocMathReal
    *   The `real` op returns the real part of a complex number.
    *
    *   Given a tensor `input` of potentially complex numbers, the op returns a tensor of type `FLOAT32` or `FLOAT64` 
    *   that is the real part of each element in `input`. If `input` contains complex numbers of the form `a + bj`, 
    *   *a* is the real part returned by the op and *b* is the imaginary part.
    *
    *   For example:
    *   {{{
    *     // 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *     real(input) ==> [-2.25, 3.25]
    *   }}}
    *
    *   Note that, if `input` is already real-valued, then it is returned unchanged.
    * 
    * @define OpDocMathImag
    *   The `imag` op returns the real part of a complex number.
    *
    *   Given a tensor `input` of complex numbers, the op returns a tensor of type `FLOAT32` or `FLOAT64` that is the
    *   imaginary part of each element in `input`. If `input` contains complex numbers of the form `a + bj`, *a* is the
    *   real part and *b* is the imaginary part returned by the op.
    *
    *   For example:
    *   {{{
    *     // 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *     real(input) ==> [4.75, 5.75]
    *   }}}
    * 
    * @define OpDocMathAngle
    *   The `angle` op returns the element-wise complex argument of a tensor.
    *
    *   Given a numeric tensor `input`, the op returns a tensor with numbers that are the complex angle of each element 
    *   in `input`. If the numbers in `input` are of the form `a + bj`, where *a* is the real part and *b* is the
    *   imaginary part, then the complex angle returned by this operation is of the form `atan2(b, a)`.
    *
    *   For example:
    *   {{{
    *     // 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *     angle(input) ==> [2.0132, 1.056]
    *   }}}
    *
    *   If `input` is real-valued, then a tensor containing zeros is returned.
    * 
    * @define OpDocMathConjugate
    *   The `conjugate` op returns the element-wise complex conjugate of a tensor.
    *
    *   Given a numeric tensor `input`, the op returns a tensor with numbers that are the complex conjugate of each
    *   element in `input`. If the numbers in `input` are of the form `a + bj`, where *a* is the real part and *b* is 
    *   the imaginary part, then the complex conjugate returned by this operation is of the form `a - bj`.
    *
    *   For example:
    *   {{{
    *     // 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    *     conjugate(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
    *   }}}
    *
    *   If `input` is real-valued, then it is returned unchanged.
    * 
    * @define OpDocMathBucketize
    *   The `bucketize` op bucketizes a tensor based on the provided boundaries.
    *
    *   For example:
    *   {{{
    *     // 'input' is [[-5, 10000], [150, 10], [5, 100]]
    *     // 'boundaries' are [0, 10, 100]
    *     bucketize(input, boundaries) ==> [[0, 3], [3, 2], [1, 3]]
    *   }}}
    * 
    * @define OpDocMathZerosFraction
    *   The `zerosFraction` op computes the fraction of zeros in `input`.
    *
    *   If `input` is empty, the result is `NaN`.
    *
    *   This is useful in summaries to measure and report sparsity.
    */
  private[ops] trait Documentation
}
