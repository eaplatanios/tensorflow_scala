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

package org.platanios.tensorflow.api.tensors.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.InvalidArgumentException
import org.platanios.tensorflow.jni.generated.tensors.{Math => NativeTensorOpsMath}

import scala.util.DynamicVariable

/** Contains functions for executing general math-related ops.
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
    * @return Result as a new tensor.
    */
  def select(condition: Tensor, x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.select(
      executionContext.value.nativeHandle, condition.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathRange
    *
    * @group MathOps
    * @param  start Rank 0 (i.e., scalar) tensor that contains the starting value of the number sequence.
    * @param  limit Rank 0 (i.e., scalar) tensor that contains the ending value (exclusive) of the number sequence.
    * @param  delta Rank 0 (i.e., scalar) tensor that contains the difference between consecutive numbers in the
    *               sequence.
    * @return Result as a new tensor.
    */
  def range(
      start: Tensor,
      limit: Tensor,
      delta: Tensor = 1,
      dataType: DataType = null
  ): Tensor = {
    var castedStart: Tensor = start
    var castedLimit: Tensor = limit
    var castedDelta: Tensor = delta
    val inferredDataType = {
      if (dataType != null)
        dataType
      else
        DataType.mostPrecise(start.dataType, limit.dataType, delta.dataType)
    }
    if (start.dataType != inferredDataType)
      castedStart = cast(start, inferredDataType)
    if (limit.dataType != inferredDataType)
      castedLimit = cast(limit, inferredDataType)
    if (delta.dataType != inferredDataType)
      castedDelta = cast(delta, inferredDataType)
    Tensor.fromNativeHandle(
      NativeTensorOpsMath.range(
        executionContext.value.nativeHandle, castedStart.nativeHandle, castedLimit.nativeHandle,
        castedDelta.nativeHandle))
  }

  /** $OpDocMathLinspace
    *
    * @group MathOps
    * @param  start          Rank 0 (i.e., scalar) tensor that contains the starting value of the number sequence.
    * @param  stop           Rank 0 (i.e., scalar) tensor that contains the ending value (inclusive) of the number
    *                        sequence.
    * @param  numberOfValues Rank 0 (i.e., scalar) tensor that contains the number of values in the number sequence.
    * @return Result as a new tensor.
    */
  def linspace(
      start: Tensor,
      stop: Tensor,
      numberOfValues: Tensor
  ): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.linSpace(
      executionContext.value.nativeHandle, start.nativeHandle, stop.nativeHandle, numberOfValues.nativeHandle))
  }

  /** $OpDocMathCast
    *
    * @group MathOps
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @return Result as a new tensor.
    */
  def cast[T <: TensorLike : TensorOps](x: T, dataType: DataType): T = {
    if (x.dataType == dataType) {
      x
    } else {
      implicitly[TensorOps[T]]
          .applyUnary(x, t =>
            Tensor.fromNativeHandle(NativeTensorOpsMath.cast(
              executionContext.value.nativeHandle, t.nativeHandle, dataType.cValue)))
    }
  }

  // TODO: [OPS] saturateCast

  /** $OpDocMathBitcast
    *
    * @group MathOps
    * @param  input    Input tensor.
    * @param  dataType Target data type.
    * @return Result as a new tensor.
    */
  def bitcast(input: Tensor, dataType: DataType): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.bitcast(
      executionContext.value.nativeHandle, input.nativeHandle, dataType.cValue))
  }

  /** $OpDocMathAddN
    *
    * @group MathOps
    * @param  inputs Input tensors.
    * @return Result as a new tensor.
    */
  def addN(inputs: Seq[Tensor]): Tensor = {
    if (inputs.length == 1)
      inputs.head
    else
      Tensor.fromNativeHandle(NativeTensorOpsMath.addN(
        executionContext.value.nativeHandle, castArgs(inputs).map(_.nativeHandle).toArray))
  }

  // TODO: [OPS] accumulateN

  //region Unary Ops

  /** $OpDocMathAbs
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def abs[T <: TensorLike : TensorOps](x: T): T = {
    if (x.dataType.isComplex) {
      implicitly[TensorOps[T]]
          .applyUnary(x, t =>
            Tensor.fromNativeHandle(NativeTensorOpsMath.complexAbs(
              executionContext.value.nativeHandle, t.nativeHandle, x.dataType.cValue)))
    } else {
      implicitly[TensorOps[T]]
          .applyUnary(x, t =>
            Tensor.fromNativeHandle(NativeTensorOpsMath.abs(executionContext.value.nativeHandle, t.nativeHandle)))
    }
  }

  /** $OpDocMathNegate
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def negate[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.neg(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathReciprocal
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def reciprocal[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.reciprocal(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathSquare
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def square[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.square(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathSqrt
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def sqrt[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.sqrt(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathRsqrt
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def rsqrt[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.rsqrt(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathExp
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def exp[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.exp(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathExpm1
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def expm1[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.expm1(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathLog
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def log[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.log(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathLog1p
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def log1p[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.log1p(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathSin
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def sin[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.sin(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathCos
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def cos[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.cos(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathTan
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def tan[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.tan(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathAsin
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def asin[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.asin(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathAcos
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def acos[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.acos(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathAtan
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def atan[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.atan(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathSinh
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def sinh[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.sinh(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathCosh
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def cosh[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.cosh(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathTanh
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def tanh[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.tanh(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathAsinh
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def asinh[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.asinh(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathAcosh
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def acosh[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.acosh(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathAtanh
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def atanh[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.atanh(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathLogGamma
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def logGamma[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.lgamma(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathDigamma
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def digamma[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.digamma(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathErf
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def erf[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.erf(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathErfc
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def erfc[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.erfc(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathSigmoid
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def sigmoid[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.sigmoid(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  // TODO: [OPS] logSigmoid

  /** $OpDocMathSign
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, `INT64`,
    *           `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def sign[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.sign(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathRound
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `COMPLEX64`, or
    *           `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def round[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.round(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathRoundInt
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def roundInt[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.rint(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathFloor
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def floor[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.floor(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathCeil
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def ceil[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.ceil(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathIsNaN
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def isNaN[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.isNan(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathIsInf
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def isInf[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.isInf(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  /** $OpDocMathIsFinite
    *
    * @group MathOps
    * @param  x Input tensor that must be one of the following types: `HALF`, `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def isFinite[T: TensorOps](x: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(x, t =>
          Tensor.fromNativeHandle(NativeTensorOpsMath.isFinite(executionContext.value.nativeHandle, t.nativeHandle)))
  }

  //endregion Unary Ops

  //region Binary Ops

  /** $OpDocMathAdd
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, `COMPLEX128`, or `STRING`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, `COMPLEX128`, or `STRING`.
    * @return Result as a new tensor.
    */
  def add(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.add(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathSubtract
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *           `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *           `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def subtract(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.sub(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathMultiply
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def multiply(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.mul(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathDivide
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def divide(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.div(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathFloorDivide
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  @deprecated("Use `truncateDivide` instead.", "0.1")
  def floorDivide(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.floorDiv(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathTruncateDivide
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def truncateDivide(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.truncateDiv(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathRealDivide
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `UINT8`,
    *           `INT8`, `INT16`, `INT32`, `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def realDivide(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.realDiv(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathSquaredDifference
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *           `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *           `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def squaredDifference(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.squaredDifference(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathMod
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or `INT64`.
    * @param  y Second input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or `INT64`.
    * @return Result as a new tensor.
    */
  def mod(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.mod(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathFloorMod
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or `INT64`.
    * @param  y Second input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or `INT64`.
    * @return Result as a new tensor.
    */
  def floorMod(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.floorMod(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathTruncateMod
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or `INT64`.
    * @param  y Second input tensor that must be one of the following types: `FLOAT32`, `FLOAT64`, `INT32`, or `INT64`.
    * @return Result as a new tensor.
    */
  def truncateMod(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.truncateMod(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathPow
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *           `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`,
    *           `INT64`, `COMPLEX64`, or `COMPLEX128`.
    * @return Result as a new tensor.
    */
  def pow(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.pow(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathIgammac
    *
    * @group MathOps
    * @param  a First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def igammac(a: Tensor, x: Tensor): Tensor = {
    val (cA, cX) = castArgs(a, x)
    Tensor.fromNativeHandle(NativeTensorOpsMath.igammac(
      executionContext.value.nativeHandle, cA.nativeHandle, cX.nativeHandle))
  }

  /** $OpDocMathIgamma
    *
    * @group MathOps
    * @param  a First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def igamma(a: Tensor, x: Tensor): Tensor = {
    val (cA, cX) = castArgs(a, x)
    Tensor.fromNativeHandle(NativeTensorOpsMath.igamma(
      executionContext.value.nativeHandle, cA.nativeHandle, cX.nativeHandle))
  }

  /** $OpDocMathZeta
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  q Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def zeta(x: Tensor, q: Tensor): Tensor = {
    val (cX, cQ) = castArgs(x, q)
    Tensor.fromNativeHandle(NativeTensorOpsMath.zeta(
      executionContext.value.nativeHandle, cX.nativeHandle, cQ.nativeHandle))
  }

  /** $OpDocMathPolygamma
    *
    * @group MathOps
    * @param  n First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def polygamma(n: Tensor, x: Tensor): Tensor = {
    val (cN, cX) = castArgs(n, x)
    Tensor.fromNativeHandle(NativeTensorOpsMath.polygamma(
      executionContext.value.nativeHandle, cN.nativeHandle, cX.nativeHandle))
  }

  /** $OpDocMathAtan2
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  y Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def atan2(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.atan2(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathMaximum
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, or
    *           `INT64`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, or 
    *           `INT64`.
    * @return Result as a new tensor.
    */
  def maximum(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.maximum(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathMinimum
    *
    * @group MathOps
    * @param  x First input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, or
    *           `INT64`.
    * @param  y Second input tensor that must be one of the following types: `HALF`, `FLOAT32`, `FLOAT64`, `INT32`, or 
    *           `INT64`.
    * @return Result as a new tensor.
    */
  def minimum(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.minimum(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  //endregion Binary Ops

  /** $OpDocMathIncompleteBeta
    *
    * @group MathOps
    * @param  a First input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  b Second input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @param  x Third input tensor that must be one of the following types: `FLOAT32`, or `FLOAT64`.
    * @return Result as a new tensor.
    */
  def incompleteBeta(a: Tensor, b: Tensor, x: Tensor): Tensor = {
    val (cA, cB, cX) = castArgs(a, b, x)
    Tensor.fromNativeHandle(NativeTensorOpsMath.betainc(
      executionContext.value.nativeHandle, cA.nativeHandle, cB.nativeHandle, cX.nativeHandle))
  }

  //region Logical Ops

  /** $OpDocMathLogicalNot
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def logicalNot(x: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.logicalNot(
      executionContext.value.nativeHandle, x.nativeHandle))
  }

  /** $OpDocMathLogicalAnd
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def logicalAnd(x: Tensor, y: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.logicalAnd(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathLogicalOr
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def logicalOr(x: Tensor, y: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.logicalOr(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathLogicalXOr
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def logicalXOr(x: Tensor, y: Tensor): Tensor = {
    logicalAnd(logicalOr(x, y), logicalNot(logicalAnd(x, y)))
  }

  //endregion Logical Ops

  //region Comparison Ops

  /** $OpDocMathEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def equal(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.equal(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathNotEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def notEqual(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.notEqual(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** $OpDocMathApproximatelyEqual
    *
    * @group MathOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  tolerance Comparison tolerance value.
    * @return Result as a new tensor.
    */
  def approximatelyEqual(
      x: Tensor,
      y: Tensor,
      tolerance: Float = 0.00001f
  ): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.approximateEqual(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle, tolerance))
  }

  /** OpDocMathLess
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def less(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.less(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** OpDocMathLessEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def lessEqual(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.lessEqual(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** OpDocMathGreater
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def greater(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.greater(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  /** OpDocMathGreaterEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def greaterEqual(x: Tensor, y: Tensor): Tensor = {
    val (cX, cY) = castArgs(x, y)
    Tensor.fromNativeHandle(NativeTensorOpsMath.greaterEqual(
      executionContext.value.nativeHandle, cX.nativeHandle, cY.nativeHandle))
  }

  //endregion Comparison Ops

  //region Reduction Ops

  private[this] def reductionAxes[T <: TensorLike](tensor: T, axes: Tensor): Tensor = {
    if (axes != null) {
      axes
    } else {
      tensor match { // Fast path: Avoid creating range and rank ops if the rank is known statically.
        case t: Tensor if t.rank > -1 => 0 until t.rank
        // case t: TensorIndexedSlices if t.denseShape.shape.isFullyDefined =>
        //   Basic.constant(0 until t.denseShape.shape(0))
        // case t: SparseTensor if t.denseShape.shape.isFullyDefined =>
        //   Basic.constant(0 until t.denseShape.shape(0))
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
    * @return Result as a new tensor.
    */
  def sum(input: Tensor, axes: Tensor = null, keepDims: Boolean = false): Tensor = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle(NativeTensorOpsMath.sum(
        executionContext.value.nativeHandle, input.nativeHandle, reductionAxes(input, axes).nativeHandle, keepDims))
  }

  /** $OpDocMathMean
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def mean(input: Tensor, axes: Tensor = null, keepDims: Boolean = false): Tensor = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle(NativeTensorOpsMath.mean(
        executionContext.value.nativeHandle, input.nativeHandle, reductionAxes(input, axes).nativeHandle, keepDims))
  }

  /** $OpDocMathProd
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def prod(input: Tensor, axes: Tensor = null, keepDims: Boolean = false): Tensor = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle(NativeTensorOpsMath.prod(
        executionContext.value.nativeHandle, input.nativeHandle, reductionAxes(input, axes).nativeHandle, keepDims))
  }

  /** $OpDocMathMin
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def min(input: Tensor, axes: Tensor = null, keepDims: Boolean = false): Tensor = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle(NativeTensorOpsMath.min(
        executionContext.value.nativeHandle, input.nativeHandle, reductionAxes(input, axes).nativeHandle, keepDims))
  }

  /** $OpDocMathMax
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def max(input: Tensor, axes: Tensor = null, keepDims: Boolean = false): Tensor = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle(NativeTensorOpsMath.max(
        executionContext.value.nativeHandle, input.nativeHandle, reductionAxes(input, axes).nativeHandle, keepDims))
  }

  /** $OpDocMathAll
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def all(input: Tensor, axes: Tensor = null, keepDims: Boolean = false): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.all(
      executionContext.value.nativeHandle, input.nativeHandle, reductionAxes(input, axes).nativeHandle, keepDims))
  }

  /** $OpDocMathAny
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def any(input: Tensor, axes: Tensor = null, keepDims: Boolean = false): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.any(
      executionContext.value.nativeHandle, input.nativeHandle, reductionAxes(input, axes).nativeHandle, keepDims))
  }

  /** $OpDocMathLogSumExp
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer sequence containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def logSumExp(input: Tensor, axes: Seq[Int] = null, keepDims: Boolean = false): Tensor = {
    if (input.rank == 0) {
      input
    } else {
      val axesTensor: Tensor = axes
      val maxValue = Basic.stopGradient(max(input, axesTensor, keepDims = true))
      val result = add(log(sum(exp(input - maxValue), axesTensor, keepDims = true)), maxValue)
      if (keepDims)
        result
      else
        Basic.squeeze(result, axes)
    }
  }

  /** $OpDocMathCountNonZero
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer array containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor with [[INT64]] data type.
    */
  def countNonZero(input: Tensor, axes: Tensor = null, keepDims: Boolean = false): Tensor = {
    sum(cast(notEqual(input, Tensor.zeros(input.dataType, Shape())), INT64), axes, keepDims)
  }

  //endregion Reduction Ops

  /** $OpDocMathArgmax
    *
    * @group MathOps
    * @param  input          Input tensor.
    * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor. Must be `INT32` or `INT64`.
    * @return Result as a new tensor.
    */
  def argmax(input: Tensor, axes: Tensor = 0, outputDataType: DataType = INT64): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.argMax(
      executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, outputDataType.cValue))
  }

  /** $OpDocMathArgmin
    *
    * @group MathOps
    * @param  input          Input tensor.
    * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor. Must be [[INT32]] or [[INT64]].
    * @return Result as a new tensor.
    */
  def argmin(input: Tensor, axes: Tensor = 0, outputDataType: DataType = INT64): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.argMin(
      executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, outputDataType.cValue))
  }

  /** $OpDocMathBinCount
    *
    * @group MathOps
    * @param  input     `INT32` tensor containing non-negative values.
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
      input: Tensor,
      weights: Tensor = null,
      minLength: Tensor = null,
      maxLength: Tensor = null,
      dataType: DataType = INT32
  ): Tensor = {
    val inputNonEmpty = greater(prod(Basic.shape(input)), 0)
    var outputSize = cast(inputNonEmpty, INT32) * add(max(input), 1)
    if (minLength != null)
      outputSize = maximum(minLength, outputSize)
    if (maxLength != null)
      outputSize = minimum(maxLength, outputSize)
    val effectiveWeights = {
      if (weights != null) {
        weights
      } else {
        Tensor.zeros(dataType, Shape.scalar())
      }
    }
    Tensor.fromNativeHandle(NativeTensorOpsMath.bincount(
      executionContext.value.nativeHandle, input.nativeHandle, outputSize.nativeHandle, effectiveWeights.nativeHandle))
  }

  /** $OpDocMathCumsum
    *
    * @group MathOps
    * @param  input     Input tensor.
    * @param  axis      [[INT32]] tensor containing the axis along which to perform the cumulative sum.
    * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
    * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
    * @return Result as a new tensor.
    */
  def cumsum(
      input: Tensor,
      axis: Tensor = 0,
      exclusive: Boolean = false,
      reverse: Boolean = false
  ): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.cumsum(
      executionContext.value.nativeHandle, input.nativeHandle, axis.nativeHandle, exclusive, reverse))
  }

  /** $OpDocMathCumprod
    *
    * @group MathOps
    * @param  input     Input tensor.
    * @param  axis      `INT32` tensor containing the axis along which to perform the cumulative product.
    * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative product.
    * @param  reverse   Boolean value indicating whether to perform a reverse cumulative product.
    * @return Result as a new tensor.
    */
  def cumprod(
      input: Tensor,
      axis: Tensor = 0,
      exclusive: Boolean = false,
      reverse: Boolean = false
  ): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.cumprod(
      executionContext.value.nativeHandle, input.nativeHandle, axis.nativeHandle, exclusive, reverse))
  }

  //region Segment Ops

  /** $OpDocMathSegmentSum
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentSum(data: Tensor, segmentIndices: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.segmentSum(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentMean
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentMean(data: Tensor, segmentIndices: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.segmentMean(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentProd
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentProd(data: Tensor, segmentIndices: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.segmentProd(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentMin
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentMin(data: Tensor, segmentIndices: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.segmentMin(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentMax
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentMax(data: Tensor, segmentIndices: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.segmentMax(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathUnsortedSegmentSum
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
    * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
    * @return Result as a new tensor.
    */
  def unsortedSegmentSum(data: Tensor, segmentIndices: Tensor, segmentsNumber: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.unsortedSegmentSum(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle, segmentsNumber.nativeHandle))
  }

  /** $OpDocMathUnsortedSegmentMax
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
    * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
    * @return Result as a new tensor.
    */
  def unsortedSegmentMax(data: Tensor, segmentIndices: Tensor, segmentsNumber: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.unsortedSegmentMax(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle, segmentsNumber.nativeHandle))
  }

  /** $OpDocMathSparseSegmentSum
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  numSegments    Optional `INT32` scalar indicating the size of the output tensor.
    * @return Result as a new tensor.
    */
  def sparseSegmentSum(data: Tensor, indices: Tensor, segmentIndices: Tensor, numSegments: Tensor = null): Tensor = {
    if (numSegments == null)
      Tensor.fromNativeHandle(NativeTensorOpsMath.sparseSegmentSum(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle))
    else
      Tensor.fromNativeHandle(NativeTensorOpsMath.sparseSegmentSumWithNumSegments(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle,
        numSegments.nativeHandle))
  }

  /** $OpDocMathSparseSegmentMean
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  numSegments    Optional `INT32` scalar indicating the size of the output tensor.
    * @return Result as a new tensor.
    */
  def sparseSegmentMean(data: Tensor, indices: Tensor, segmentIndices: Tensor, numSegments: Tensor = null): Tensor = {
    if (numSegments == null)
      Tensor.fromNativeHandle(NativeTensorOpsMath.sparseSegmentMean(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle))
    else
      Tensor.fromNativeHandle(NativeTensorOpsMath.sparseSegmentMeanWithNumSegments(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle,
        numSegments.nativeHandle))
  }

  /** $OpDocMathSparseSegmentSumSqrtN
    *
    * @group MathOps
    *
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
    *                        and can be repeated.
    * @param  numSegments    Optional `INT32` scalar indicating the size of the output tensor.
    * @return Result as a new tensor.
    */
  def sparseSegmentSumSqrtN(
      data: Tensor,
      indices: Tensor,
      segmentIndices: Tensor,
      numSegments: Tensor = null
  ): Tensor = {
    if (numSegments == null)
      Tensor.fromNativeHandle(NativeTensorOpsMath.sparseSegmentSqrtN(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle))
    else
      Tensor.fromNativeHandle(NativeTensorOpsMath.sparseSegmentSqrtNWithNumSegments(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle,
        numSegments.nativeHandle))
  }

  //endregion Segment Ops

  //region Matrix Ops

  /** $OpDocMathDiag
    *
    * @group MathOps
    *
    * @param  diagonal Diagonal values, represented as a rank-`K` tensor, where `K` can be at most `3`.
    * @return Result as a new tensor.
    */
  def diag(diagonal: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.diag(executionContext.value.nativeHandle, diagonal.nativeHandle))
  }

  /** $OpDocMathDiagPart
    *
    * @group MathOps
    *
    * @param  input Rank-`K` input tensor, where `K` is either `2`, `4`, or `6`.
    * @return Result as a new tensor.
    */
  def diagPart(input: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.diagPart(executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocMathMatrixDiag
    *
    * @group MathOps
    *
    * @param  diagonal Rank-`K` input tensor, where `K >= 1`.
    * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its 
    *         last dimension duplicated.
    */
  def matrixDiag(diagonal: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.matrixDiag(executionContext.value.nativeHandle, diagonal.nativeHandle))
  }

  /** $OpDocMathMatrixSetDiag
    *
    * @group MathOps
    *
    * @param  input    Rank-`K+1` tensor, where `K >= 2`.
    * @param  diagonal Rank-`K` tensor, where `K >= 1`.
    * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `input`.
    */
  def matrixSetDiag(input: Tensor, diagonal: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.matrixSetDiag(
      executionContext.value.nativeHandle, input.nativeHandle, diagonal.nativeHandle))
  }

  /** $OpDocMathMatrixDiagPart
    *
    * @group MathOps
    *
    * @param  input Rank-`K` tensor, where `K >= 2`.
    * @return Result as a new tensor containing the diagonal(s) and having shape equal to
    *         `input.shape[:-2] + [min(input.shape[-2:])]`.
    */
  def matrixDiagPart(input: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.matrixDiagPart(executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocMathMatrixBandPart
    *
    * @group MathOps
    *
    * @param  input             Input tensor.
    * @param  numSubDiagonals   Scalar `INT64` tensor that contains the number of sub-diagonals to keep. If negative,
    *                           the entire lower triangle is kept.
    * @param  numSuperDiagonals Scalar `INT64` tensor that contains the number of super-diagonals to keep. If negative,
    *                           the entire upper triangle is kept.
    * @return Result as a new tensor containing the expected banded tensor and has rank `K` and same shape as `input`.
    */
  def matrixBandPart(input: Tensor, numSubDiagonals: Tensor, numSuperDiagonals: Tensor): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.matrixBandPart(
      executionContext.value.nativeHandle, input.nativeHandle, numSubDiagonals.nativeHandle,
      numSuperDiagonals.nativeHandle))
  }

  /** $OpDocMathTrace
    *
    * @group MathOps
    *
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def trace(input: Tensor): Tensor = {
    sum(matrixDiagPart(input), axes = -1)
  }

  /** $OpDocMathScalarMul
    *
    * @group MathOps
    *
    * @param  scalar Scalar tensor.
    * @param  tensor Tensor to multiply the scalar tensor with.
    * @return Result as a new tensor.
    */
  def scalarMul[T: TensorOps](scalar: Tensor, tensor: T): T = {
    implicitly[TensorOps[T]].applyUnary(tensor, t => multiply(scalar, t))
  }

  /** $OpDocMathMatmul
    *
    * @group MathOps
    *
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
    * @return Result as a new tensor.
    */
  def matmul(
      a: Tensor,
      b: Tensor,
      transposeA: Boolean = false,
      transposeB: Boolean = false,
      conjugateA: Boolean = false,
      conjugateB: Boolean = false,
      aIsSparse: Boolean = false,
      bIsSparse: Boolean = false
  ): Tensor = {
    val (cA, cB) = castArgs(a, b)
    val sparseMatMulDataTypes = Set[DataType](BFLOAT16, FLOAT32)
    if (!aIsSparse && !bIsSparse && (cA.rank == -1 || cA.rank > 2) && (cB.rank == -1 || cB.rank > 2)) {
      // "BatchMatMul" does not support transpose, so we conjugate the matrix and use adjoint instead.
      // The "conj" op is a no-op for real matrices.
      val (x, adjointX) = transposeConjugateToAdjoint(cA, transposeA, conjugateA)
      val (y, adjointY) = transposeConjugateToAdjoint(cB, transposeB, conjugateB)
      Tensor.fromNativeHandle(NativeTensorOpsMath.batchMatMul(
        executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, adjointX, adjointY))
    } else if (cA.dataType == BFLOAT16 || cB.dataType == BFLOAT16 || // "MatMul" does not currently support this type.
        ((aIsSparse || bIsSparse) &&
            sparseMatMulDataTypes.contains(cA.dataType) &&
            sparseMatMulDataTypes.contains(cB.dataType))) {
      val (x, transposeX) = transposeConjugateToTranspose(cA, transposeA, conjugateA)
      val (y, transposeY) = transposeConjugateToTranspose(cB, transposeB, conjugateB)
      Tensor.fromNativeHandle(NativeTensorOpsMath.sparseMatMul(
        executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, transposeX, transposeY,
        aIsSparse, bIsSparse))
    } else {
      val (x, transposeX) = transposeConjugateToTranspose(cA, transposeA, conjugateA)
      val (y, transposeY) = transposeConjugateToTranspose(cB, transposeB, conjugateB)
      Tensor.fromNativeHandle(NativeTensorOpsMath.matMul(
        executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, transposeX, transposeY))
    }
  }

  private[this] def transposeConjugateToAdjoint(
      tensor: Tensor, transpose: Boolean, conj: Boolean): (Tensor, Boolean) = {
    (transpose, conj) match {
      case (false, false) => (tensor, false)
      case (false, true) => (conjugate(tensor), false)
      case (true, false) => (conjugate(tensor), true)
      case (true, true) => (tensor, true)
    }
  }

  private[this] def transposeConjugateToTranspose(
      tensor: Tensor, transpose: Boolean, conj: Boolean): (Tensor, Boolean) = {
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
    *
    * @param  a First input tensor.
    * @param  b Second input tensor.
    * @return Result as a new tensor.
    */
  def cross(a: Tensor, b: Tensor): Tensor = {
    val (cA, cB) = castArgs(a, b)
    Tensor.fromNativeHandle(NativeTensorOpsMath.cross(
      executionContext.value.nativeHandle, cA.nativeHandle, cB.nativeHandle))
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
  def tensorDot(a: Tensor, b: Tensor, numAxes: Tensor): Tensor = {
    if (numAxes.rank != 0)
      throw InvalidArgumentException("'numAxes' must be a scalar.")
    tensorDot(a, b, range(a.rank - numAxes, a.rank), range(0, numAxes))
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
  def tensorDot(a: Tensor, b: Tensor, axesA: Tensor, axesB: Tensor): Tensor = {
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
    def tensorDotReshape(a: Tensor, axes: Tensor, flipped: Boolean = false): (Tensor, Tensor) = {
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

    val (reshapedA, freeA) = tensorDotReshape(a, axesA)
    val (reshapedB, freeB) = tensorDotReshape(b, axesB, flipped = true)
    val abMatmul = matmul(reshapedA, reshapedB)
    Basic.reshape(abMatmul, Basic.concatenate(Seq(freeA, freeB), 0))
  }

  //endregion Matrix Ops

  //region Complex Ops

  /** $OpDocMathComplex
    *
    * @group MathOps
    *
    * @param  real Tensor containing the real component. Must have [[FLOAT32]] or [[FLOAT64]] data type.
    * @param  imag Tensor containing the imaginary component. Must have [[FLOAT32]] or [[FLOAT64]] data type.
    * @return Result as a new tensor with data type being either [[COMPLEX64]] or [[COMPLEX128]].
    * @throws IllegalArgumentException If 'real' and 'imag' have invalid data types.
    */
  @throws[IllegalArgumentException]
  def complex(real: Tensor, imag: Tensor): Tensor = {
    val (cReal, cImag) = castArgs(real, imag)
    val outputDataType = (cReal.dataType, cImag.dataType) match {
      case (FLOAT32, FLOAT32) => COMPLEX64
      case (FLOAT64, FLOAT64) => COMPLEX128
      case _ => throw new IllegalArgumentException(
        s"'real' (dataType = ${real.dataType}) and 'imag' (dataType = ${imag.dataType}) must both have the same data " +
            s"type, which must be either 'FLOAT32' or 'FLOAT64'.")
    }
    Tensor.fromNativeHandle(NativeTensorOpsMath.complex(
      executionContext.value.nativeHandle, cReal.nativeHandle, cImag.nativeHandle, outputDataType.cValue))
  }

  /** $OpDocMathReal
    *
    * @group MathOps
    *
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def real[T <: TensorLike : TensorOps](input: T): T = {
    if (!input.dataType.isComplex) {
      input
    } else {
      implicitly[TensorOps[T]]
          .applyUnary(input, t =>
            Tensor.fromNativeHandle(NativeTensorOpsMath.real(
              executionContext.value.nativeHandle, t.nativeHandle, t.dataType.real.cValue)))
    }
  }

  /** $OpDocMathImag
    *
    * @group MathOps
    *
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def imag[T <: TensorLike : TensorOps](input: T): T = {
    if (!input.dataType.isComplex) {
      input
    } else {
      implicitly[TensorOps[T]]
          .applyUnary(input, t =>
            Tensor.fromNativeHandle(NativeTensorOpsMath.imag(
              executionContext.value.nativeHandle, t.nativeHandle, t.dataType.real.cValue)))
    }
  }

  /** $OpDocMathAngle
    *
    * @group MathOps
    *
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def angle[T <: TensorLike : TensorOps](input: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(input, t => {
          if (t.dataType.isComplex) {
            Tensor.fromNativeHandle(NativeTensorOpsMath.angle(
              executionContext.value.nativeHandle, t.nativeHandle, t.dataType.real.cValue))
          } else if (t.dataType.isNumeric) {
            Tensor.zeros(t.dataType, t.shape)
          } else {
            throw new IllegalArgumentException("'angle' can only take numeric tensors as input.")
          }
        })
  }

  /** $OpDocMathConjugate
    *
    * @group MathOps
    *
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def conjugate[T <: TensorLike : TensorOps](input: T): T = {
    implicitly[TensorOps[T]]
        .applyUnary(input, t => {
          if (t.dataType.isComplex) {
            Tensor.fromNativeHandle(NativeTensorOpsMath.conj(executionContext.value.nativeHandle, t.nativeHandle))
          } else if (t.dataType.isNumeric) {
            t
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
    *
    * @param  input      Numeric tensor to bucketize.
    * @param  boundaries Sorted sequence of `Float`s specifying the boundaries of the buckets.
    * @return Result as a new tensor.
    */
  def bucketize(input: Tensor, boundaries: Seq[Float]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.bucketize(
      executionContext.value.nativeHandle, input.nativeHandle, boundaries.toArray))
  }

  //endregion Bucketization Ops

  //region Other Ops

  /** $OpDocMathZerosFraction
    *
    * @group MathOps
    *
    * @param  input Input tensor.
    * @return Result as a new tensor, with `FLOAT32` data type.
    */
  def zerosFraction(input: Tensor): Tensor = {
    mean(cast(equal(input, Tensor.fill(input.dataType)(0)), FLOAT32))
  }

  //endregion Other Ops
}

object Math extends Math {
  case class MathOps(tensor: Tensor) {
    //region Operators

    /** $OpDocMathNegate
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def unary_- : Tensor = negate

    /** $OpDocMathAdd
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def +(other: Tensor): Tensor = add(other)

    /** $OpDocMathSubtract
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def -(other: Tensor): Tensor = subtract(other)

    /** $OpDocMathMultiply
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def *(other: Tensor): Tensor = multiply(other)

    private[this] def divHelper(x: Tensor, y: Tensor): Tensor = {
      if (x.dataType.isFloatingPoint || x.dataType.isComplex)
        Math.divide(x, y)
      else
        Math.truncateDivide(x, y)
    }

    /** $OpDocMathDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def /(other: Tensor): Tensor = divHelper(tensor, other)

    /** $OpDocMathMod
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def %(other: Tensor): Tensor = mod(other)

    /** $OpDocMathPow
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def **(other: Tensor): Tensor = pow(other)

    /** $OpDocMathPow
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def ^(other: Tensor): Tensor = pow(other)

    /** $OpDocMathLogicalNot
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def unary_! : Tensor = logicalNot

    /** $OpDocMathLogicalAnd
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def &&(other: Tensor): Tensor = logicalAnd(other)

    /** $OpDocMathLogicalOr
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def ||(other: Tensor): Tensor = logicalOr(other)

    /** $OpDocMathEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def ==(other: Tensor): Tensor = equal(other)

    /** $OpDocMathNotEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def !=(other: Tensor): Tensor = notEqual(other)

    /** $OpDocMathLess
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def <(other: Tensor): Tensor = less(other)

    /** $OpDocMathLessEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def <=(other: Tensor): Tensor = lessEqual(other)

    /** $OpDocMathGreater
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def >(other: Tensor): Tensor = greater(other)

    /** $OpDocMathGreaterEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def >=(other: Tensor): Tensor = greaterEqual(other)

    //endregion Operators

    /** $OpDocMathCast
      *
      * @group MathOps
      * @param  dataType Target data type.
      * @return Result as a new tensor.
      */
    def cast(dataType: DataType): Tensor = Math.cast(tensor, dataType)

    /** $OpDocMathBitcast
      *
      * @group MathOps
      * @param  dataType Target data type.
      * @return Result as a new tensor.
      */
    def bitcast(dataType: DataType): Tensor = Math.bitcast(tensor, dataType)

    //region Unary Ops

    /** $OpDocMathAbs
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def abs: Tensor = Math.abs(tensor)

    /** $OpDocMathNegate
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def negate: Tensor = Math.negate(tensor)

    /** $OpDocMathReciprocal
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def reciprocal: Tensor = Math.reciprocal(tensor)

    /** $OpDocMathSquare
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def square: Tensor = Math.square(tensor)

    /** $OpDocMathSqrt
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sqrt: Tensor = Math.sqrt(tensor)

    /** $OpDocMathRsqrt
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def rsqrt: Tensor = Math.rsqrt(tensor)

    /** $OpDocMathExp
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def exp: Tensor = Math.exp(tensor)

    /** $OpDocMathExpm1
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def expm1: Tensor = Math.expm1(tensor)

    /** $OpDocMathLog
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def log: Tensor = Math.log(tensor)

    /** $OpDocMathLog1p
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def log1p: Tensor = Math.log1p(tensor)

    /** $OpDocMathSin
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sin: Tensor = Math.sin(tensor)

    /** $OpDocMathCos
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def cos: Tensor = Math.cos(tensor)

    /** $OpDocMathTan
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def tan: Tensor = Math.tan(tensor)

    /** $OpDocMathAsin
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def asin: Tensor = Math.asin(tensor)

    /** $OpDocMathAcos
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def acos: Tensor = Math.acos(tensor)

    /** $OpDocMathAtan
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def atan: Tensor = Math.atan(tensor)

    /** $OpDocMathSinh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sinh: Tensor = Math.sinh(tensor)

    /** $OpDocMathCosh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def cosh: Tensor = Math.cosh(tensor)

    /** $OpDocMathTanh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def tanh: Tensor = Math.tanh(tensor)

    /** $OpDocMathAsinh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def asinh: Tensor = Math.asinh(tensor)

    /** $OpDocMathAcosh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def acosh: Tensor = Math.acosh(tensor)

    /** $OpDocMathAtanh
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def atanh: Tensor = Math.atanh(tensor)

    /** $OpDocMathLogGamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logGamma: Tensor = Math.logGamma(tensor)

    /** $OpDocMathDigamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def digamma: Tensor = Math.digamma(tensor)

    /** $OpDocMathErf
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def erf: Tensor = Math.erf(tensor)

    /** $OpDocMathErfc
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def erc: Tensor = Math.erfc(tensor)

    /** $OpDocMathSigmoid
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sigmoid: Tensor = Math.sigmoid(tensor)

    /** $OpDocMathSign
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def sign: Tensor = Math.sign(tensor)

    /** $OpDocMathRound
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def round: Tensor = Math.round(tensor)

    /** $OpDocMathRoundInt
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def roundInt: Tensor = Math.roundInt(tensor)

    /** $OpDocMathFloor
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def floor: Tensor = Math.floor(tensor)

    /** $OpDocMathCeil
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def ceil: Tensor = Math.ceil(tensor)

    /** $OpDocMathIsNaN
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def isNaN: Tensor = Math.isNaN(tensor)

    /** $OpDocMathIsInf
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def isInf: Tensor = Math.isInf(tensor)

    /** $OpDocMathIsFinite
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def isFinite: Tensor = Math.isFinite(tensor)

    //endregion Unary Ops

    //region Binary Ops

    /** $OpDocMathAdd
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def add(other: Tensor): Tensor = Math.add(tensor, other)

    /** $OpDocMathSubtract
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def subtract(other: Tensor): Tensor = Math.subtract(tensor, other)

    /** $OpDocMathMultiply
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def multiply(other: Tensor): Tensor = Math.multiply(tensor, other)

    /** $OpDocMathDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def divide(other: Tensor): Tensor = Math.divide(tensor, other)

    /** $OpDocMathFloorDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    @deprecated("Use `truncateDivide` instead.", "0.1")
    def floorDivide(other: Tensor): Tensor = Math.floorDivide(tensor, other)

    /** $OpDocMathTruncateDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def truncateDivide(other: Tensor): Tensor = Math.truncateDivide(tensor, other)

    /** $OpDocMathRealDivide
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def realDivide(other: Tensor): Tensor = Math.realDivide(tensor, other)

    /** $OpDocMathSquaredDifference
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def squaredDifference(other: Tensor): Tensor = Math.squaredDifference(tensor, other)

    /** $OpDocMathMod
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def mod(other: Tensor): Tensor = Math.mod(tensor, other)

    /** $OpDocMathFloorMod
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def floorMod(other: Tensor): Tensor = Math.floorMod(tensor, other)

    /** $OpDocMathTruncateMod
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def truncateMod(other: Tensor): Tensor = Math.truncateMod(tensor, other)

    /** $OpDocMathPow
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def pow(other: Tensor): Tensor = Math.pow(tensor, other)

    /** $OpDocMathIgammac
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def igammac(other: Tensor): Tensor = Math.igammac(tensor, other)

    /** $OpDocMathIgamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def igamma(other: Tensor): Tensor = Math.igamma(tensor, other)

    /** $OpDocMathZeta
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def zeta(other: Tensor): Tensor = Math.zeta(tensor, other)

    /** $OpDocMathPolygamma
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def polygamma(other: Tensor): Tensor = Math.polygamma(tensor, other)

    /** $OpDocMathAtan2
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def atan2(other: Tensor): Tensor = Math.atan2(tensor, other)

    /** $OpDocMathMaximum
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def maximum(other: Tensor): Tensor = Math.maximum(tensor, other)

    /** $OpDocMathMinimum
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def minimum(other: Tensor): Tensor = Math.minimum(tensor, other)

    //endregion Binary Ops

    //region Logical Ops

    /** $OpDocMathLogicalNot
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logicalNot: Tensor = Math.logicalNot(tensor)

    /** $OpDocMathLogicalAnd
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logicalAnd(other: Tensor): Tensor = Math.logicalAnd(tensor, other)

    /** $OpDocMathLogicalOr
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logicalOr(other: Tensor): Tensor = Math.logicalOr(tensor, other)

    /** $OpDocMathLogicalXOr
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def logicalXOr(other: Tensor): Tensor = Math.logicalXOr(tensor, other)

    //endregion Logical Ops

    //region Comparison Ops

    /** $OpDocMathEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def equal(other: Tensor): Tensor = Math.equal(tensor, other)

    /** $OpDocMathNotEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def notEqual(other: Tensor): Tensor = Math.notEqual(tensor, other)

    /** $OpDocMathApproximatelyEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def approximatelyEqual(other: Tensor): Tensor = Math.approximatelyEqual(tensor, other)

    /** $OpDocMathLess
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def less(other: Tensor): Tensor = Math.less(tensor, other)

    /** $OpDocMathLessEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def lessEqual(other: Tensor): Tensor = Math.lessEqual(tensor, other)

    /** $OpDocMathGreater
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def greater(other: Tensor): Tensor = Math.greater(tensor, other)

    /** $OpDocMathGreaterEqual
      *
      * @group MathOps
      * @return Result as a new tensor.
      */
    def greaterEqual(other: Tensor): Tensor = Math.greaterEqual(tensor, other)

    //endregion Comparison Ops

    //region Reduction Ops

    /** $OpDocMathSum
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def sum(axes: Tensor = null, keepDims: Boolean = false): Tensor = Math.sum(tensor, axes, keepDims)

    /** $OpDocMathMean
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def mean(axes: Tensor = null, keepDims: Boolean = false): Tensor = Math.mean(tensor, axes, keepDims)

    /** $OpDocMathProd
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def prod(axes: Tensor = null, keepDims: Boolean = false): Tensor = Math.prod(tensor, axes, keepDims)

    /** $OpDocMathMin
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def min(axes: Tensor = null, keepDims: Boolean = false): Tensor = Math.min(tensor, axes, keepDims)

    /** $OpDocMathMax
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def max(axes: Tensor = null, keepDims: Boolean = false): Tensor = Math.max(tensor, axes, keepDims)

    /** $OpDocMathAll
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def all(axes: Tensor = null, keepDims: Boolean = false): Tensor = Math.all(tensor, axes, keepDims)

    /** $OpDocMathAny
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def any(axes: Tensor = null, keepDims: Boolean = false): Tensor = Math.any(tensor, axes, keepDims)

    /** $OpDocMathLogSumExp
      *
      * @group MathOps
      * @param  axes     Integer sequence containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def logSumExp(axes: Seq[Int] = null, keepDims: Boolean = false): Tensor = Math.logSumExp(tensor, axes, keepDims)

    /** $OpDocMathCountNonZero
      *
      * @group MathOps
      * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Result as a new tensor.
      */
    def countNonZero(axes: Tensor = null, keepDims: Boolean = false): Tensor = Math.countNonZero(tensor, axes, keepDims)

    //endregion Reduction Ops

    /** $OpDocMathArgmax
      *
      * @group MathOps
      * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  outputDataType Data type for the output tensor. Must be `INT32` or `INT64`.
      * @return Result as a new tensor.
      */
    def argmax(axes: Tensor = 0, outputDataType: DataType = INT64): Tensor = Math.argmax(tensor, axes, outputDataType)

    /** $OpDocMathArgmin
      *
      * @group MathOps
      * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
      * @param  outputDataType Data type for the output tensor. Must be `INT32` or `INT64`.
      * @return Result as a new tensor.
      */
    def argmin(axes: Tensor = 0, outputDataType: DataType = INT64): Tensor = Math.argmin(tensor, axes, outputDataType)

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
        weights: Tensor = null, minLength: Tensor = null, maxLength: Tensor = null,
        dataType: DataType = INT32): Tensor = {
      Math.binCount(tensor, weights, minLength, maxLength, dataType)
    }

    /** $OpDocMathCumsum
      *
      * @group MathOps
      * @param  axis      [[INT32]] tensor containing the axis along which to perform the cumulative sum.
      * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
      * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
      * @return Result as a new tensor.
      */
    def cumsum(axis: Tensor = 0, exclusive: Boolean = false, reverse: Boolean = false): Tensor = {
      Math.cumsum(tensor, axis, exclusive, reverse)
    }

    /** $OpDocMathCumprod
      *
      * @group MathOps
      * @param  axis      [[INT32]] tensor containing the axis along which to perform the cumulative product.
      * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative product.
      * @param  reverse   Boolean value indicating whether to perform a reverse cumulative product.
      * @return Result as a new tensor.
      */
    def cumprod(axis: Tensor = 0, exclusive: Boolean = false, reverse: Boolean = false): Tensor = {
      Math.cumprod(tensor, axis, exclusive, reverse)
    }

    //region Segment Ops

    /** $OpDocMathSegmentSum
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentSum(segmentIndices: Tensor): Tensor = Math.segmentSum(tensor, segmentIndices)

    /** $OpDocMathSegmentMean
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentMean(segmentIndices: Tensor): Tensor = Math.segmentMean(tensor, segmentIndices)

    /** $OpDocMathSegmentProd
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentProd(segmentIndices: Tensor): Tensor = Math.segmentProd(tensor, segmentIndices)

    /** $OpDocMathSegmentMin
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentMin(segmentIndices: Tensor): Tensor = Math.segmentMin(tensor, segmentIndices)

    /** $OpDocMathSegmentMax
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]). Values should be sorted
      *                        and can be repeated.
      * @return Result as a new tensor.
      */
    def segmentMax(segmentIndices: Tensor): Tensor = Math.segmentMax(tensor, segmentIndices)

    /** $OpDocMathUnsortedSegmentSum
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
      * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
      * @return Result as a new tensor.
      */
    def unsortedSegmentSum(segmentIndices: Tensor, segmentsNumber: Tensor): Tensor = {
      Math.unsortedSegmentSum(tensor, segmentIndices, segmentsNumber)
    }

    /** $OpDocMathUnsortedSegmentMax
      *
      * @group MathOps
      *
      * @param  segmentIndices Segment indices (must have data type of [[INT32]] or [[INT64]]).
      * @param  segmentsNumber Number of segments (must have data type of [[INT32]]).
      * @return Result as a new tensor.
      */
    def unsortedSegmentMax(segmentIndices: Tensor, segmentsNumber: Tensor): Tensor = {
      Math.unsortedSegmentMax(tensor, segmentIndices, segmentsNumber)
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
    def sparseSegmentSum(indices: Tensor, segmentIndices: Tensor, numSegments: Tensor = null): Tensor = {
      Math.sparseSegmentSum(tensor, indices, segmentIndices, numSegments)
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
    def sparseSegmentMean(indices: Tensor, segmentIndices: Tensor, numSegments: Tensor = null): Tensor = {
      Math.sparseSegmentMean(tensor, indices, segmentIndices, numSegments)
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
    def sparseSegmentSumSqrtN(indices: Tensor, segmentIndices: Tensor, numSegments: Tensor = null): Tensor = {
      Math.sparseSegmentSumSqrtN(tensor, indices, segmentIndices, numSegments)
    }

    //endregion Segment Ops

    //region Matrix Ops

    /** $OpDocMathDiag
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def diag: Tensor = Math.diag(tensor)

    /** $OpDocMathDiagPart
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def diagPart: Tensor = Math.diagPart(tensor)

    /** $OpDocMathMatrixDiag
      *
      * @group MathOps
      *
      * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its
      *         last dimension duplicated.
      */
    def matrixDiag: Tensor = Math.matrixDiag(tensor)

    /** $OpDocMathMatrixSetDiag
      *
      * @group MathOps
      *
      * @param  diagonal Rank-`K` tensor, where `K >= 1`.
      * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `input`.
      */
    def matrixSetDiag(diagonal: Tensor): Tensor = Math.matrixSetDiag(tensor, diagonal)

    /** $OpDocMathMatrixDiagPart
      *
      * @group MathOps
      *
      * @return Result as a new tensor containing the diagonal(s) and having shape equal to
      *         `input.shape[:-2] + [min(input.shape[-2:])]`.
      */
    def matrixDiagPart: Tensor = Math.matrixDiagPart(tensor)

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
    def matrixBandPart(numSubDiagonals: Tensor, numSuperDiagonals: Tensor): Tensor = {
      Math.matrixBandPart(tensor, numSubDiagonals, numSuperDiagonals)
    }

    /** $OpDocMathTrace
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def trace: Tensor = Math.trace(tensor)

    /** $OpDocMathMatmul
      *
      * @group MathOps
      *
      * @param  other      Tensor to multiply with, with data type one of: `BFLOAT16`, `FLOAT16`, `FLOAT32`, `FLOAT64`,
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
        other: Tensor, transposeA: Boolean = false, transposeB: Boolean = false, conjugateA: Boolean = false,
        conjugateB: Boolean = false, aIsSparse: Boolean = false, bIsSparse: Boolean = false): Tensor = {
      Math.matmul(tensor, other, transposeA, transposeB, conjugateA, conjugateB, aIsSparse, bIsSparse)
    }

    /** $OpDocMathCross
      *
      * @group MathOps
      * @param  other Tensor to multiply with.
      *
      * @return Result as a new tensor.
      */
    def cross(other: Tensor): Tensor = Math.cross(tensor, other)

    /** Dynamic version (i.e., where `numAxes` may be a symbolic tensor) of the `tensorDot` op.
      *
      * $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other   Tensor to contract with.
      * @param  numAxes Number of axes to contract.
      * @return Created op output.
      */
    def tensorDot(other: Tensor, numAxes: Tensor): Tensor = {
      Math.tensorDot(tensor, other, numAxes)
    }

    /** Dynamic version (i.e., where `axesA` and `axesB` may be symbolic tensors) of the `tensorDot` op.
      *
      * $OpDocMathTensorDot
      *
      * @group MathOps
      * @param  other Tensor to contract with.
      * @param  axesA Axes to contract in `a`.
      * @param  axesB Axes to contract in `b`.
      * @return Created op output.
      */
    def tensorDot(other: Tensor, axesA: Tensor, axesB: Tensor): Tensor = {
      Math.tensorDot(tensor, other, axesA, axesB)
    }

    //endregion Matrix Ops

    //region Complex Ops

    /** $OpDocMathReal
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def real: Tensor = Math.real(tensor)

    /** $OpDocMathImag
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def imag: Tensor = Math.imag(tensor)

    /** $OpDocMathAngle
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def angle: Tensor = Math.angle(tensor)

    /** $OpDocMathConjugate
      *
      * @group MathOps
      *
      * @return Result as a new tensor.
      */
    def conjugate: Tensor = Math.conjugate(tensor)

    //endregion Complex Ops

    //region Quantization Ops

    // TODO: [OPS] quantization

    //endregion Quantization Ops

    //region Bucketization Ops

    /** $OpDocMathBucketize
      *
      * @group MathOps
      *
      * @param  boundaries Sorted sequence of `Float`s specifying the boundaries of the buckets.
      * @return Result as a new tensor.
      */
    def bucketize(boundaries: Seq[Float]): Tensor = Math.bucketize(tensor, boundaries)

    //endregion Bucketization Ops

    //region Other Ops

    /** $OpDocMathZerosFraction
      *
      * @group MathOps
      *
      * @return Result as a new tensor, with `FLOAT32` data type.
      */
    def zerosFraction: Tensor = Math.zerosFraction(tensor)

    //endregion Other Ops
  }
}
