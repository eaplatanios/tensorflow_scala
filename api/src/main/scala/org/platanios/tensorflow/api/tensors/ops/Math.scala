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
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.generated.tensors.{Math => NativeTensorOpsMath}

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
  def select[D <: DataType](condition: Tensor[BOOLEAN], x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.select(
      executionContext.value.nativeHandle, condition.nativeHandle, x.nativeHandle, y.nativeHandle))
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
  def range[D <: NumericDataType](
      start: Tensor[D],
      limit: Tensor[D],
      delta: Tensor[D] = null
  ): Tensor[D] = {
    val deltaWithDefault = if (delta == null) Tensor.ones(start.dataType, Shape()) else delta
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.range(
      executionContext.value.nativeHandle, start.nativeHandle, limit.nativeHandle,
      deltaWithDefault.nativeHandle))
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
  def linspace[D <: BFloat16OrFloat32OrFloat64, I <: Int32OrInt64](
      start: Tensor[D],
      stop: Tensor[D],
      numberOfValues: Tensor[I]
  ): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.linSpace(
      executionContext.value.nativeHandle, start.nativeHandle, stop.nativeHandle, numberOfValues.nativeHandle))
  }

  /** $OpDocMathCast
    *
    * @group MathOps
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @return Result as a new tensor.
    */
  def cast[D <: DataType, DR <: DataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D], dataType: DR)(implicit
    ev: TensorOps.Aux[TL, D]
  ): TL[DR] = {
    if (x.dataType == dataType) {
      x.asInstanceOf[TL[DR]]
    } else {
      ev.applyUnary(x, t => {
        Tensor.fromNativeHandle(NativeTensorOpsMath.cast(
          executionContext.value.nativeHandle, t.nativeHandle, dataType.cValue))
      })
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
  def bitcast[D <: ReducibleDataType, DR <: DataType](input: Tensor[D], dataType: DR): Tensor[DR] = {
    Tensor.fromNativeHandle[DR](NativeTensorOpsMath.bitcast(
      executionContext.value.nativeHandle, input.nativeHandle, dataType.cValue))
  }

  /** $OpDocMathAddN
    *
    * @group MathOps
    * @param  inputs Input tensors.
    * @return Result as a new tensor.
    */
  def addN[D <: ReducibleDataType](inputs: Seq[Tensor[D]]): Tensor[D] = {
    if (inputs.length == 1)
      inputs.head
    else
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.addN(
        executionContext.value.nativeHandle, inputs.map(_.nativeHandle).toArray))
  }

  // TODO: [OPS] accumulateN

  //region Unary Ops

  /** $OpDocMathAbs
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def abs[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    if (x.dataType.isComplex) {
      ev.applyUnary(x, t => {
        Tensor.fromNativeHandle[D](NativeTensorOpsMath.complexAbs(
          executionContext.value.nativeHandle, t.nativeHandle, x.dataType.cValue))
      })
    } else {
      ev.applyUnary(x, t => {
        Tensor.fromNativeHandle[D](NativeTensorOpsMath.abs(executionContext.value.nativeHandle, t.nativeHandle))
      })
    }
  }

  /** $OpDocMathNegate
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def negate[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.neg(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathReciprocal
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def reciprocal[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.reciprocal(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSquare
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def square[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.square(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSqrt
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sqrt[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sqrt(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathRsqrt
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def rsqrt[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.rsqrt(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathExp
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def exp[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.exp(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathExpm1
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def expm1[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.expm1(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathLog
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def log[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.log(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathLog1p
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def log1p[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.log1p(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSin
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sin[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sin(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathCos
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def cos[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.cos(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathTan
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def tan[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.tan(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAsin
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def asin[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.asin(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAcos
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def acos[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.acos(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAtan
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def atan[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.atan(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSinh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sinh[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sinh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathCosh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def cosh[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.cosh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathTanh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def tanh[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.tanh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAsinh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def asinh[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.asinh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAcosh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def acosh[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.acosh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAtanh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def atanh[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.atanh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathLogGamma
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def logGamma[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.lgamma(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathDigamma
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def digamma[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.digamma(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathErf
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def erf[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.erf(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathErfc
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def erfc[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.erfc(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSigmoid
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sigmoid[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sigmoid(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  // TODO: [OPS] logSigmoid

  /** $OpDocMathSign
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sign[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sign(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathRound
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def round[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D])(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.round(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathRoundInt
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def roundInt[D <: Float16OrFloat32OrFloat64, TL[DD <: DataType] <: TensorLike[DD]](
      x: TL[D]
  )(implicit ev: TensorOps.Aux[TL, D]): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.rint(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathFloor
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def floor[D <: Float16OrFloat32OrFloat64, TL[DD <: DataType] <: TensorLike[DD]](
      x: TL[D]
  )(implicit ev: TensorOps.Aux[TL, D]): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.floor(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathCeil
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def ceil[D <: Float16OrFloat32OrFloat64, TL[DD <: DataType] <: TensorLike[DD]](
      x: TL[D]
  )(implicit ev: TensorOps.Aux[TL, D]): TL[D] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.ceil(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathIsNaN
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def isNaN[D <: Float16OrFloat32OrFloat64, TL[DD <: DataType] <: TensorLike[DD]](
      x: TL[D]
  )(implicit ev: TensorOps.Aux[TL, D]): TL[BOOLEAN] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.isNan(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathIsInf
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def isInf[D <: Float16OrFloat32OrFloat64, TL[DD <: DataType] <: TensorLike[DD]](
      x: TL[D]
  )(implicit ev: TensorOps.Aux[TL, D]): TL[BOOLEAN] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.isInf(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathIsFinite
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def isFinite[D <: Float16OrFloat32OrFloat64, TL[DD <: DataType] <: TensorLike[DD]](
      x: TL[D]
  )(implicit ev: TensorOps.Aux[TL, D]): TL[BOOLEAN] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.isFinite(
        executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  //endregion Unary Ops

  //region Binary Ops

  /** $OpDocMathAdd
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def add[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.add(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathSubtract
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def subtract[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.sub(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathMultiply
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def multiply[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.mul(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathDivide
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def divide[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.div(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathFloorDivide
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  @deprecated("Use `truncateDivide` instead.", "0.1")
  def floorDivide[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.floorDiv(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathTruncateDivide
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def truncateDivide[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.truncateDiv(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathRealDivide
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def realDivide[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.realDiv(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathSquaredDifference
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def squaredDifference[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.squaredDifference(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathMod
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def mod[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.mod(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathFloorMod
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def floorMod[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.floorMod(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathTruncateMod
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def truncateMod[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.truncateMod(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathPow
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def pow[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.pow(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  // TODO: !!! [TYPES] Fix this.
  
  /** $OpDocMathIgammac
    *
    * @group MathOps
    * @param  a First input tensor.
    * @param  x Second input tensor.
    * @return Result as a new tensor.
    */
  def igammac[D <: Float32OrFloat64](a: Tensor[D], x: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.igammac(
      executionContext.value.nativeHandle, a.nativeHandle, x.nativeHandle))
  }

  /** $OpDocMathIgamma
    *
    * @group MathOps
    * @param  a First input tensor.
    * @param  x Second input tensor.
    * @return Result as a new tensor.
    */
  def igamma[D <: Float32OrFloat64](a: Tensor[D], x: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.igamma(
      executionContext.value.nativeHandle, a.nativeHandle, x.nativeHandle))
  }

  /** $OpDocMathZeta
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  q Second input tensor.
    * @return Result as a new tensor.
    */
  def zeta[D <: Float32OrFloat64](x: Tensor[D], q: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.zeta(
      executionContext.value.nativeHandle, x.nativeHandle, q.nativeHandle))
  }

  /** $OpDocMathPolygamma
    *
    * @group MathOps
    * @param  n First input tensor.
    * @param  x Second input tensor.
    * @return Result as a new tensor.
    */
  def polygamma[D <: Float32OrFloat64](n: Tensor[D], x: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.polygamma(
      executionContext.value.nativeHandle, n.nativeHandle, x.nativeHandle))
  }

  /** $OpDocMathAtan2
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def atan2[D <: Float32OrFloat64](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.atan2(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathMaximum
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def maximum[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.maximum(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathMinimum
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def minimum[D <: MathDataType](x: Tensor[D], y: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.minimum(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  //endregion Binary Ops

  /** $OpDocMathIncompleteBeta
    *
    * @group MathOps
    * @param  a First input tensor.
    * @param  b Second input tensor.
    * @param  x Third input tensor.
    * @return Result as a new tensor.
    */
  def incompleteBeta[D <: Float32OrFloat64](a: Tensor[D], b: Tensor[D], x: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.betainc(
      executionContext.value.nativeHandle, a.nativeHandle, b.nativeHandle, x.nativeHandle))
  }

  //region Logical Ops

  /** $OpDocMathLogicalNot
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def logicalNot(x: Tensor[BOOLEAN]): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.logicalNot(
      executionContext.value.nativeHandle, x.nativeHandle))
  }

  /** $OpDocMathLogicalAnd
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def logicalAnd(x: Tensor[BOOLEAN], y: Tensor[BOOLEAN]): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.logicalAnd(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathLogicalOr
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def logicalOr(x: Tensor[BOOLEAN], y: Tensor[BOOLEAN]): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.logicalOr(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathLogicalXOr
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def logicalXOr(x: Tensor[BOOLEAN], y: Tensor[BOOLEAN]): Tensor[BOOLEAN] = {
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
  def equal[D <: ReducibleDataType](x: Tensor[D], y: Tensor[D]): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.equal(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathNotEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def notEqual[D <: ReducibleDataType](x: Tensor[D], y: Tensor[D]): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.notEqual(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathApproximatelyEqual
    *
    * @group MathOps
    * @param  x         First input tensor.
    * @param  y         Second input tensor.
    * @param  tolerance Comparison tolerance value.
    * @return Result as a new tensor.
    */
  def approximatelyEqual[D <: ReducibleDataType](
      x: Tensor[D],
      y: Tensor[D],
      tolerance: Float = 0.00001f
  ): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.approximateEqual(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, tolerance))
  }

  /** OpDocMathLess
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def less[D <: ReducibleDataType](x: Tensor[D], y: Tensor[D]): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.less(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** OpDocMathLessEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def lessEqual[D <: ReducibleDataType](x: Tensor[D], y: Tensor[D]): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.lessEqual(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** OpDocMathGreater
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def greater[D <: ReducibleDataType](x: Tensor[D], y: Tensor[D]): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.greater(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** OpDocMathGreaterEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def greaterEqual[D <: ReducibleDataType](x: Tensor[D], y: Tensor[D]): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.greaterEqual(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  //endregion Comparison Ops

  //region Reduction Ops

  private[this] def reductionAxes[T <: TensorLike[_]](
      tensor: T,
      axes: Tensor[INT32]
  ): Tensor[INT32] = {
    if (axes != null) {
      axes
    } else {
      tensor match { // Fast path: Avoid creating range and rank ops if the rank is known statically.
        case t: Tensor[_] if t.rank > -1 => Tensor(0 until t.rank: _*)
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
  def sum[D <: ReducibleDataType](
      input: Tensor[D],
      axes: Tensor[INT32] = null,
      keepDims: Boolean = false
  ): Tensor[D] = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sum(
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
  def mean[D <: ReducibleDataType](
      input: Tensor[D],
      axes: Tensor[INT32] = null,
      keepDims: Boolean = false
  ): Tensor[D] = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.mean(
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
  def prod[D <: ReducibleDataType](
      input: Tensor[D],
      axes: Tensor[INT32] = null,
      keepDims: Boolean = false
  ): Tensor[D] = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.prod(
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
  def min[D <: ReducibleDataType](
      input: Tensor[D],
      axes: Tensor[INT32] = null,
      keepDims: Boolean = false
  ): Tensor[D] = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.min(
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
  def max[D <: ReducibleDataType](
      input: Tensor[D],
      axes: Tensor[INT32] = null,
      keepDims: Boolean = false
  ): Tensor[D] = {
    if (input.rank == 0)
      input
    else
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.max(
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
  def all(input: Tensor[BOOLEAN], axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.all(
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
  def any(input: Tensor[BOOLEAN], axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[BOOLEAN] = {
    Tensor.fromNativeHandle[BOOLEAN](NativeTensorOpsMath.any(
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
  def logSumExp[D <: MathDataType](input: Tensor[D], axes: Seq[Int] = null, keepDims: Boolean = false): Tensor[D] = {
    if (input.rank == 0) {
      input
    } else {
      val axesTensor: Tensor[INT32] = axes
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
    * @return Result as a new tensor.
    */
  def countNonZero[D <: ReducibleDataType](
      input: Tensor[D],
      axes: Tensor[INT32] = null,
      keepDims: Boolean = false
  ): Tensor[INT64] = {
    sum(cast(notEqual(input, Tensor.zeros(input.dataType, Shape())), INT64), axes, keepDims)
  }

  //endregion Reduction Ops

  /** $OpDocMathArgmax
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  axes  Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @return Result as a new tensor.
    */
  def argmax[D <: MathDataType, I <: Int32OrInt64](input: Tensor[D], axes: Tensor[I]): Tensor[INT64] = {
    argmax(input, axes, INT64)
  }

  /** $OpDocMathArgmax
    *
    * @group MathOps
    * @param  input          Input tensor.
    * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor.
    * @return Result as a new tensor.
    */
  def argmax[D <: MathDataType, I <: Int32OrInt64, IR <: Int32OrInt64](
      input: Tensor[D],
      axes: Tensor[I],
      outputDataType: IR
  ): Tensor[IR] = {
    Tensor.fromNativeHandle[IR](NativeTensorOpsMath.argMax(
      executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, outputDataType.cValue))
  }

  /** $OpDocMathArgmin
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  axes  Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @return Result as a new tensor.
    */
  def argmin[D <: MathDataType, I <: Int32OrInt64](input: Tensor[D], axes: Tensor[I]): Tensor[INT64] = {
    argmin(input, axes, INT64)
  }

  /** $OpDocMathArgmin
    *
    * @group MathOps
    * @param  input          Input tensor.
    * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  outputDataType Data type for the output tensor.
    * @return Result as a new tensor.
    */
  def argmin[D <: MathDataType, I <: Int32OrInt64, IR <: Int32OrInt64](
      input: Tensor[D],
      axes: Tensor[I],
      outputDataType: IR
  ): Tensor[IR] = {
    Tensor.fromNativeHandle[IR](NativeTensorOpsMath.argMin(
      executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, outputDataType.cValue))
  }

  /** $OpDocMathBinCount
    *
    * @group MathOps
    * @param  input     Tensor containing non-negative values.
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
  def binCount[D <: Int32OrInt64OrFloat32OrFloat64](
      input: Tensor[INT32],
      weights: Tensor[D] = null,
      minLength: Tensor[INT32] = null,
      maxLength: Tensor[INT32] = null,
      dataType: D = null
  ): Tensor[D] = {
    val inputNonEmpty = greater(prod(Basic.shape(input)), 0)
    var outputSize = cast(inputNonEmpty, INT32) * add(max(input), Tensor.ones(INT32, Shape()))
    if (minLength != null)
      outputSize = maximum(minLength, outputSize)
    if (maxLength != null)
      outputSize = minimum(maxLength, outputSize)
    val effectiveWeights = {
      if (weights != null) {
        weights
      } else if (dataType == null) {
        Tensor.zeros(INT32, Shape.scalar())
      } else {
        Tensor.zeros(dataType, Shape.scalar())
      }
    }
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.bincount(
      executionContext.value.nativeHandle, input.nativeHandle, outputSize.nativeHandle, effectiveWeights.nativeHandle))
  }

  /** $OpDocMathCumsum
    *
    * @group MathOps
    * @param  input     Input tensor.
    * @param  axis      Tensor containing the axis along which to perform the cumulative sum.
    * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
    * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
    * @return Result as a new tensor.
    */
  def cumsum[D <: MathDataType](
      input: Tensor[D],
      axis: Tensor[INT32] = 0,
      exclusive: Boolean = false,
      reverse: Boolean = false
  ): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.cumsum(
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
  def cumprod[D <: MathDataType](
      input: Tensor[D],
      axis: Tensor[INT32] = 0,
      exclusive: Boolean = false,
      reverse: Boolean = false
  ): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.cumprod(
      executionContext.value.nativeHandle, input.nativeHandle, axis.nativeHandle, exclusive, reverse))
  }

  //region Segment Ops

  /** $OpDocMathSegmentSum
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentSum[D <: ReducibleDataType, I <: Int32OrInt64](data: Tensor[D], segmentIndices: Tensor[I]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.segmentSum(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentMean
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentMean[D <: ReducibleDataType, I <: Int32OrInt64](data: Tensor[D], segmentIndices: Tensor[I]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.segmentMean(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentProd
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentProd[D <: ReducibleDataType, I <: Int32OrInt64](data: Tensor[D], segmentIndices: Tensor[I]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.segmentProd(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentMin
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentMin[D <: ReducibleDataType, I <: Int32OrInt64](data: Tensor[D], segmentIndices: Tensor[I]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.segmentMin(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentMax
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentMax[D <: ReducibleDataType, I <: Int32OrInt64](data: Tensor[D], segmentIndices: Tensor[I]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.segmentMax(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathUnsortedSegmentSum
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @return Result as a new tensor.
    */
  def unsortedSegmentSum[D <: DataType, I <: Int32OrInt64](
      data: Tensor[D],
      segmentIndices: Tensor[I],
      segmentsNumber: Tensor[INT32]
  ): Tensor[D] = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.unsortedSegmentSum(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle, segmentsNumber.nativeHandle))
  }

  /** $OpDocMathUnsortedSegmentMax
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @return Result as a new tensor.
    */
  def unsortedSegmentMax[D <: DataType, I <: Int32OrInt64](
      data: Tensor[D],
      segmentIndices: Tensor[I],
      segmentsNumber: Tensor[INT32]
  ): Tensor[D] = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.unsortedSegmentMax(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle, segmentsNumber.nativeHandle))
  }

  /** $OpDocMathSparseSegmentSum
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  numSegments    Optional scalar indicating the size of the output tensor.
    * @return Result as a new tensor.
    */
  def sparseSegmentSum[D <: DataType, I1 <: Int32OrInt64, I2 <: Int32OrInt64](
      data: Tensor[D],
      indices: Tensor[I1],
      segmentIndices: Tensor[I2],
      numSegments: Tensor[INT32] = null
  ): Tensor[D] = {
    if (numSegments == null)
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sparseSegmentSum(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle))
    else
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sparseSegmentSumWithNumSegments(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle,
        numSegments.nativeHandle))
  }

  /** $OpDocMathSparseSegmentMean
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  numSegments    Optional scalar indicating the size of the output tensor.
    * @return Result as a new tensor.
    */
  def sparseSegmentMean[D <: DataType, I1 <: Int32OrInt64, I2 <: Int32OrInt64](
      data: Tensor[D],
      indices: Tensor[I1],
      segmentIndices: Tensor[I2],
      numSegments: Tensor[INT32] = null
  ): Tensor[D] = {
    if (numSegments == null)
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sparseSegmentMean(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle))
    else
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sparseSegmentMeanWithNumSegments(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle,
        numSegments.nativeHandle))
  }

  /** $OpDocMathSparseSegmentSumSqrtN
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  numSegments    Optional scalar indicating the size of the output tensor.
    * @return Result as a new tensor.
    */
  def sparseSegmentSumSqrtN[D <: DataType, I1 <: Int32OrInt64, I2 <: Int32OrInt64](
      data: Tensor[D],
      indices: Tensor[I1],
      segmentIndices: Tensor[I2],
      numSegments: Tensor[INT32] = null
  ): Tensor[D] = {
    if (numSegments == null)
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sparseSegmentSqrtN(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle))
    else
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.sparseSegmentSqrtNWithNumSegments(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle,
        numSegments.nativeHandle))
  }

  //endregion Segment Ops

  //region Matrix Ops

  /** $OpDocMathDiag
    *
    * @group MathOps
    * @param  diagonal Diagonal values, represented as a rank-`K` tensor, where `K` can be at most `3`.
    * @return Result as a new tensor.
    */
  def diag[D <: MathDataType](diagonal: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.diag(
      executionContext.value.nativeHandle, diagonal.nativeHandle))
  }

  /** $OpDocMathDiagPart
    *
    * @group MathOps
    * @param  input Rank-`K` input tensor, where `K` is either `2`, `4`, or `6`.
    * @return Result as a new tensor.
    */
  def diagPart[D <: MathDataType](input: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.diagPart(
      executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocMathMatrixDiag
    *
    * @group MathOps
    * @param  diagonal Rank-`K` input tensor, where `K >= 1`.
    * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its 
    *         last dimension duplicated.
    */
  def matrixDiag[D <: MathDataType](diagonal: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.matrixDiag(
      executionContext.value.nativeHandle, diagonal.nativeHandle))
  }

  /** $OpDocMathMatrixSetDiag
    *
    * @group MathOps
    * @param  input    Rank-`K+1` tensor, where `K >= 2`.
    * @param  diagonal Rank-`K` tensor, where `K >= 1`.
    * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `input`.
    */
  def matrixSetDiag[D <: MathDataType](input: Tensor[D], diagonal: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.matrixSetDiag(
      executionContext.value.nativeHandle, input.nativeHandle, diagonal.nativeHandle))
  }

  /** $OpDocMathMatrixDiagPart
    *
    * @group MathOps
    * @param  input Rank-`K` tensor, where `K >= 2`.
    * @return Result as a new tensor containing the diagonal(s) and having shape equal to
    *         `input.shape[:-2] + [min(input.shape[-2:])]`.
    */
  def matrixDiagPart[D <: MathDataType](input: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.matrixDiagPart(
      executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocMathMatrixBandPart
    *
    * @group MathOps
    * @param  input             Input tensor.
    * @param  numSubDiagonals   Scalar tensor that contains the number of sub-diagonals to keep. If negative,
    *                           the entire lower triangle is kept.
    * @param  numSuperDiagonals Scalar tensor that contains the number of super-diagonals to keep. If negative,
    *                           the entire upper triangle is kept.
    * @return Result as a new tensor containing the expected banded tensor and has rank `K` and same shape as `input`.
    */
  def matrixBandPart[D <: MathDataType, I <: Int32OrInt64](
      input: Tensor[D],
      numSubDiagonals: Tensor[I],
      numSuperDiagonals: Tensor[I]
  ): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.matrixBandPart(
      executionContext.value.nativeHandle, input.nativeHandle, numSubDiagonals.nativeHandle,
      numSuperDiagonals.nativeHandle))
  }

  /** $OpDocMathTrace
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def trace[D <: MathDataType](input: Tensor[D]): Tensor[D] = {
    sum(matrixDiagPart(input), axes = -1)
  }

  /** $OpDocMathScalarMul
    *
    * @group MathOps
    * @param  scalar Scalar tensor.
    * @param  tensor Tensor to multiply the scalar tensor with.
    * @return Result as a new tensor.
    */
  def scalarMul[D <: MathDataType, TL[DD <: DataType] <: TensorLike[DD]](
      scalar: Tensor[D],
      tensor: TL[D]
  )(implicit ev: TensorOps.Aux[TL, D]): TL[D] = {
    ev.applyUnary(tensor, t => multiply(scalar, t))
  }

  /** $OpDocMathMatmul
    *
    * @group MathOps
    * @param  a          First input tensor.
    * @param  b          Second input tensor.
    * @param  transposeA If `true`, `a` is transposed before the multiplication.
    * @param  transposeB If `true`, `b` is transposed before the multiplication.
    * @param  conjugateA If `true`, `a` is conjugated before the multiplication.
    * @param  conjugateB If `true`, `b` is conjugated before the multiplication.
    * @param  aIsSparse  If `true`, `a` is treated as a sparse matrix (i.e., it is assumed it contains many zeros).
    * @param  bIsSparse  If `true`, `b` is treated as a sparse matrix (i.e., it is assumed it contains many zeros).
    * @return Result as a new tensor.
    */
  def matmul[D <: MathDataType](
      a: Tensor[D],
      b: Tensor[D],
      transposeA: Boolean = false,
      transposeB: Boolean = false,
      conjugateA: Boolean = false,
      conjugateB: Boolean = false,
      aIsSparse: Boolean = false,
      bIsSparse: Boolean = false
  ): Tensor[D] = {
    val sparseMatMulDataTypes = Set[DataType](BFLOAT16, FLOAT32)
    if (!aIsSparse && !bIsSparse && (a.rank == -1 || a.rank > 2) && (b.rank == -1 || b.rank > 2)) {
      // "BatchMatMul" does not support transpose, so we conjugate the matrix and use adjoint instead.
      // The "conj" op is a no-op for real matrices.
      val (x, adjointX) = transposeConjugateToAdjoint(a, transposeA, conjugateA)
      val (y, adjointY) = transposeConjugateToAdjoint(b, transposeB, conjugateB)
      Tensor.fromNativeHandle(NativeTensorOpsMath.batchMatMul(
        executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, adjointX, adjointY))
    } else if (a.dataType == BFLOAT16 || b.dataType == BFLOAT16 || // "MatMul" does not currently support this type.
        ((aIsSparse || bIsSparse) &&
            sparseMatMulDataTypes.contains(a.dataType) &&
            sparseMatMulDataTypes.contains(b.dataType))) {
      val (x, transposeX) = transposeConjugateToTranspose(a, transposeA, conjugateA)
      val (y, transposeY) = transposeConjugateToTranspose(b, transposeB, conjugateB)
      Tensor.fromNativeHandle(NativeTensorOpsMath.sparseMatMul(
        executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, transposeX, transposeY,
        aIsSparse, bIsSparse))
    } else {
      val (x, transposeX) = transposeConjugateToTranspose(a, transposeA, conjugateA)
      val (y, transposeY) = transposeConjugateToTranspose(b, transposeB, conjugateB)
      Tensor.fromNativeHandle(NativeTensorOpsMath.matMul(
        executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, transposeX, transposeY))
    }
  }

  private[this] def transposeConjugateToAdjoint[D <: MathDataType](
      tensor: Tensor[D],
      transpose: Boolean,
      conj: Boolean
  ): (Tensor[D], Boolean) = {
    // TODO: [TYPES] These runtime checks are not elegant.
    (transpose, conj) match {
      case (false, false) => (tensor, false)
      case (false, true) if tensor.isInstanceOf[Tensor[COMPLEX64]] =>
        (conjugate(tensor.asInstanceOf[Tensor[COMPLEX64]]).asInstanceOf[Tensor[D]], false)
      case (false, true) if tensor.isInstanceOf[Tensor[COMPLEX128]] =>
        (conjugate(tensor.asInstanceOf[Tensor[COMPLEX128]]).asInstanceOf[Tensor[D]], false)
      case (false, true) => (tensor, false)
      case (true, false) if tensor.isInstanceOf[Tensor[COMPLEX64]] =>
        (conjugate(tensor.asInstanceOf[Tensor[COMPLEX64]]).asInstanceOf[Tensor[D]], true)
      case (true, false) if tensor.isInstanceOf[Tensor[COMPLEX128]] =>
        (conjugate(tensor.asInstanceOf[Tensor[COMPLEX128]]).asInstanceOf[Tensor[D]], true)
      case (true, false) => (tensor, true)
      case (true, true) => (tensor, true)
    }
  }

  private[this] def transposeConjugateToTranspose[D <: MathDataType](
      tensor: Tensor[D],
      transpose: Boolean,
      conj: Boolean
  ): (Tensor[D], Boolean) = {
    // TODO: [TYPES] These runtime checks are not elegant.
    (transpose, conj) match {
      case (false, false) => (tensor, false)
      case (false, true) if tensor.isInstanceOf[Tensor[COMPLEX64]] =>
        (conjugate(tensor.asInstanceOf[Tensor[COMPLEX64]]).asInstanceOf[Tensor[D]], false)
      case (false, true) if tensor.isInstanceOf[Tensor[COMPLEX128]] =>
        (conjugate(tensor.asInstanceOf[Tensor[COMPLEX128]]).asInstanceOf[Tensor[D]], false)
      case (false, true) => (tensor, false)
      case (true, false) => (tensor, true)
      case (true, true) if tensor.isInstanceOf[Tensor[COMPLEX64]] =>
        (conjugate(tensor.asInstanceOf[Tensor[COMPLEX64]]).asInstanceOf[Tensor[D]], true)
      case (true, true) if tensor.isInstanceOf[Tensor[COMPLEX128]] =>
        (conjugate(tensor.asInstanceOf[Tensor[COMPLEX128]]).asInstanceOf[Tensor[D]], true)
      case (true, true) => (tensor, true)
    }
  }

  /** $OpDocMathCross
    *
    * @group MathOps
    * @param  a First input tensor.
    * @param  b Second input tensor.
    * @return Result as a new tensor.
    */
  def cross[D <: MathDataType](a: Tensor[D], b: Tensor[D]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.cross(
      executionContext.value.nativeHandle, a.nativeHandle, b.nativeHandle))
  }

  /** Dynamic version (i.e., where `numAxes` may be a tensor) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    * @group MathOps
    * @param  a       First tensor.
    * @param  b       Second tensor.
    * @param  numAxes Number of axes to contract.
    * @return Created op output.
    * @throws InvalidShapeException If `numAxes` is not a scalar.
    */
  @throws[InvalidShapeException]
  def tensorDot[D <: MathDataType](a: Tensor[D], b: Tensor[D], numAxes: Tensor[INT32]): Tensor[D] = {
    if (numAxes.rank != 0)
      throw InvalidShapeException("'numAxes' must be a scalar.")
    tensorDot(a, b, range(subtract(a.rank, numAxes), a.rank), range(0, numAxes))
  }

  /** Dynamic version (i.e., where `axesA` and `axesB` may be tensors) of the `tensorDot` op.
    *
    * $OpDocMathTensorDot
    *
    * @group MathOps
    * @param  a     First tensor.
    * @param  b     Second tensor.
    * @param  axesA Axes to contract in `a`.
    * @param  axesB Axes to contract in `b`.
    * @return Created op output.
    * @throws InvalidShapeException If `axesA` or `axesB` is not a scalar.
    */
  @throws[InvalidShapeException]
  def tensorDot[D <: MathDataType](
      a: Tensor[D],
      b: Tensor[D],
      axesA: Tensor[INT32],
      axesB: Tensor[INT32]
  ): Tensor[D] = {
    if (axesA.rank != 1)
      throw InvalidShapeException("'axesA' must be a vector.")
    if (axesB.rank != 1)
      throw InvalidShapeException("'axesB' must be a vector.")

    /** Helper method to perform transpose and reshape for the tensor contraction op. This method is helpful in reducing
      * `tensorDot` to `matmul` using the `transpose` and the `reshape` ops. The method takes a tensor and performs the
      * correct transpose and reshape operations for the provided indices. It returns the reshaped tensor as well as a
      * list of indices necessary to reshape the tensor back to its proper shape after the matrix multiplication.
      *
      * @param  a       Tensor being reshaped.
      * @param  axes    Sequence of unique indices of axes of `a`.
      * @param  flipped If `true`, the method assumes that `a` is the second argument in the contraction operation.
      * @return Tuple that contains: (i) the reshaped tensor `a` that allows contraction via `matmul`, and (ii) a tensor
      *         that contains the shape of the free axes.
      */
    def tensorDotReshape(a: Tensor[D], axes: Tensor[INT32], flipped: Boolean = false): (Tensor[D], Tensor[INT32]) = {
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
    * @param  real Tensor containing the real component.
    * @param  imag Tensor containing the imaginary component.
    * @return Result as a new tensor.
    */
  def complex(real: Tensor[FLOAT32], imag: Tensor[FLOAT32]): Tensor[COMPLEX64] = {
    Tensor.fromNativeHandle[COMPLEX64](NativeTensorOpsMath.complex(
      executionContext.value.nativeHandle, real.nativeHandle, imag.nativeHandle, COMPLEX64.cValue))
  }

  /** $OpDocMathComplex
    *
    * @group MathOps
    * @param  real Tensor containing the real component.
    * @param  imag Tensor containing the imaginary component.
    * @return Result as a new tensor.
    */
  def complex(real: Tensor[FLOAT64], imag: Tensor[FLOAT64]): Tensor[COMPLEX128] = {
    Tensor.fromNativeHandle[COMPLEX128](NativeTensorOpsMath.complex(
      executionContext.value.nativeHandle, real.nativeHandle, imag.nativeHandle, COMPLEX128.cValue))
  }

  /** $OpDocMathReal
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def real[TL[DD <: COMPLEX64] <: TensorLike[DD]](
      input: TL[COMPLEX64]
  )(implicit ev: TensorOps.Aux[TL, COMPLEX64]): TL[FLOAT32] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[FLOAT32](NativeTensorOpsMath.real(
        executionContext.value.nativeHandle, t.nativeHandle, t.dataType.real.cValue))
    })
  }

  /** $OpDocMathReal
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def real[TL[DD <: COMPLEX128] <: TensorLike[DD]](
      input: TL[COMPLEX128]
  )(implicit ev: TensorOps.Aux[TL, COMPLEX128]): TL[FLOAT64] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[FLOAT64](NativeTensorOpsMath.real(
        executionContext.value.nativeHandle, t.nativeHandle, t.dataType.real.cValue))
    })
  }

  /** $OpDocMathImag
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def imag[TL[DD <: COMPLEX64] <: TensorLike[DD]](
      input: TL[COMPLEX64]
  )(implicit ev: TensorOps.Aux[TL, COMPLEX64]): TL[FLOAT32] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[FLOAT32](NativeTensorOpsMath.imag(
        executionContext.value.nativeHandle, t.nativeHandle, t.dataType.real.cValue))
    })
  }

  /** $OpDocMathImag
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def imag[TL[DD <: COMPLEX128] <: TensorLike[DD]](
      input: TL[COMPLEX128]
  )(implicit ev: TensorOps.Aux[TL, COMPLEX128]): TL[FLOAT64] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[FLOAT64](NativeTensorOpsMath.imag(
        executionContext.value.nativeHandle, t.nativeHandle, t.dataType.real.cValue))
    })
  }

  /** $OpDocMathAngle
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def angle[TL[DD <: COMPLEX64] <: TensorLike[DD]](
      input: TL[COMPLEX64]
  )(implicit ev: TensorOps.Aux[TL, COMPLEX64]): TL[FLOAT32] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[FLOAT32](NativeTensorOpsMath.angle(
        executionContext.value.nativeHandle, t.nativeHandle, t.dataType.real.cValue))
    })
  }

  /** $OpDocMathAngle
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def angle[TL[DD <: COMPLEX128] <: TensorLike[DD]](
      input: TL[COMPLEX128]
  )(implicit ev: TensorOps.Aux[TL, COMPLEX128]): TL[FLOAT64] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[FLOAT64](NativeTensorOpsMath.angle(
        executionContext.value.nativeHandle, t.nativeHandle, t.dataType.real.cValue))
    })
  }

  /** $OpDocMathConjugate
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def conjugate[D <: ComplexDataType, TL[DD <: DataType] <: TensorLike[DD]](
      input: TL[D]
  )(implicit ev: TensorOps.Aux[TL, D]): TL[D] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[D](NativeTensorOpsMath.conj(executionContext.value.nativeHandle, t.nativeHandle))
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
    * @param  boundaries Sorted sequence of numbers specifying the boundaries of the buckets.
    * @return Result as a new tensor.
    */
  def bucketize[D <: Int32OrInt64OrFloat32OrFloat64](input: Tensor[D], boundaries: Seq[Float]): Tensor[D] = {
    Tensor.fromNativeHandle[D](NativeTensorOpsMath.bucketize(
      executionContext.value.nativeHandle, input.nativeHandle, boundaries.toArray))
  }

  //endregion Bucketization Ops

  //region Other Ops

  /** $OpDocMathZerosFraction
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def zerosFraction[D <: ReducibleDataType](input: Tensor[D]): Tensor[FLOAT32] = {
    mean(cast(equal(input, Tensor.fill(input.dataType, Shape())(0)), FLOAT32))
  }

  //endregion Other Ops
}

object Math extends Math {
  private[api] trait Implicits {
    implicit class MathOps[D <: DataType](val tensor: Tensor[D]) {
      /** $OpDocMathCast
        *
        * @group MathOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def cast[DR <: DataType](dataType: DR): Tensor[DR] = Math.cast(tensor, dataType)

      //region Segment Ops

      /** $OpDocMathUnsortedSegmentSum
        *
        * @group MathOps
        * @param  segmentIndices Segment indices.
        * @param  segmentsNumber Number of segments.
        * @return Result as a new tensor.
        */
      def unsortedSegmentSum[I <: Int32OrInt64](segmentIndices: Tensor[I], segmentsNumber: Tensor[INT32]): Tensor[D] = {
        Math.unsortedSegmentSum(tensor, segmentIndices, segmentsNumber)
      }

      /** $OpDocMathUnsortedSegmentMax
        *
        * @group MathOps
        * @param  segmentIndices Segment indices.
        * @param  segmentsNumber Number of segments.
        * @return Result as a new tensor.
        */
      def unsortedSegmentMax[I <: Int32OrInt64](segmentIndices: Tensor[I], segmentsNumber: Tensor[INT32]): Tensor[D] = {
        Math.unsortedSegmentMax(tensor, segmentIndices, segmentsNumber)
      }

      /** $OpDocMathSparseSegmentSum
        *
        * @group MathOps
        * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @param  numSegments    Optional scalar indicating the size of the output tensor.
        * @return Result as a new tensor.
        */
      def sparseSegmentSum[I1 <: Int32OrInt64, I2 <: Int32OrInt64](
          indices: Tensor[I1],
          segmentIndices: Tensor[I2],
          numSegments: Tensor[INT32] = null
      ): Tensor[D] = {
        Math.sparseSegmentSum(tensor, indices, segmentIndices, numSegments)
      }

      /** $OpDocMathSparseSegmentMean
        *
        * @group MathOps
        * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @param  numSegments    Optional scalar indicating the size of the output tensor.
        * @return Result as a new tensor.
        */
      def sparseSegmentMean[I1 <: Int32OrInt64, I2 <: Int32OrInt64](
          indices: Tensor[I1],
          segmentIndices: Tensor[I2],
          numSegments: Tensor[INT32] = null
      ): Tensor[D] = {
        Math.sparseSegmentMean(tensor, indices, segmentIndices, numSegments)
      }

      /** $OpDocMathSparseSegmentSumSqrtN
        *
        * @group MathOps
        * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @param  numSegments    Optional scalar indicating the size of the output tensor.
        * @return Result as a new tensor.
        */
      def sparseSegmentSumSqrtN[I1 <: Int32OrInt64, I2 <: Int32OrInt64](
          indices: Tensor[I1],
          segmentIndices: Tensor[I2],
          numSegments: Tensor[INT32] = null
      ): Tensor[D] = {
        Math.sparseSegmentSumSqrtN(tensor, indices, segmentIndices, numSegments)
      }

      //endregion Segment Ops

      //region Quantization Ops

      // TODO: [OPS] quantization

      //endregion Quantization Ops
    }

    implicit class ReducibleMathOps[D <: ReducibleDataType](val tensor: Tensor[D]) {
      /** $OpDocMathBitcast
        *
        * @group MathOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def bitcast[DR <: DataType](dataType: DR): Tensor[DR] = Math.bitcast(tensor, dataType)

      //region Operators

      /** $OpDocMathEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ==(other: Tensor[D]): Tensor[BOOLEAN] = equal(other)

      /** $OpDocMathNotEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def !=(other: Tensor[D]): Tensor[BOOLEAN] = notEqual(other)

      /** $OpDocMathLess
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def <(other: Tensor[D]): Tensor[BOOLEAN] = less(other)

      /** $OpDocMathLessEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def <=(other: Tensor[D]): Tensor[BOOLEAN] = lessEqual(other)

      /** $OpDocMathGreater
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def >(other: Tensor[D]): Tensor[BOOLEAN] = greater(other)

      /** $OpDocMathGreaterEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def >=(other: Tensor[D]): Tensor[BOOLEAN] = greaterEqual(other)

      //endregion Operators

      //region Comparison Ops

      /** $OpDocMathEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def equal(other: Tensor[D]): Tensor[BOOLEAN] = Math.equal(tensor, other)

      /** $OpDocMathNotEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def notEqual(other: Tensor[D]): Tensor[BOOLEAN] = Math.notEqual(tensor, other)

      /** $OpDocMathApproximatelyEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def approximatelyEqual(other: Tensor[D]): Tensor[BOOLEAN] = Math.approximatelyEqual(tensor, other)

      /** $OpDocMathLess
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def less(other: Tensor[D]): Tensor[BOOLEAN] = Math.less(tensor, other)

      /** $OpDocMathLessEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def lessEqual(other: Tensor[D]): Tensor[BOOLEAN] = Math.lessEqual(tensor, other)

      /** $OpDocMathGreater
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def greater(other: Tensor[D]): Tensor[BOOLEAN] = Math.greater(tensor, other)

      /** $OpDocMathGreaterEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def greaterEqual(other: Tensor[D]): Tensor[BOOLEAN] = Math.greaterEqual(tensor, other)

      //endregion Comparison Ops

      //region Reduction Ops

      /** $OpDocMathSum
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def sum(axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[D] = Math.sum(tensor, axes, keepDims)

      /** $OpDocMathMean
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def mean(axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[D] = Math.mean(tensor, axes, keepDims)

      /** $OpDocMathProd
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def prod(axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[D] = Math.prod(tensor, axes, keepDims)

      /** $OpDocMathMin
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def min(axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[D] = Math.min(tensor, axes, keepDims)

      /** $OpDocMathMax
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def max(axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[D] = Math.max(tensor, axes, keepDims)

      /** $OpDocMathCountNonZero
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def countNonZero(axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[INT64] = {
        Math.countNonZero(tensor, axes, keepDims)
      }

      //endregion Reduction Ops

      //region Segment Ops

      /** $OpDocMathSegmentSum
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentSum[I <: Int32OrInt64](segmentIndices: Tensor[I]): Tensor[D] = Math.segmentSum(tensor, segmentIndices)

      /** $OpDocMathSegmentMean
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentMean[I <: Int32OrInt64](segmentIndices: Tensor[I]): Tensor[D] = Math.segmentMean(tensor, segmentIndices)

      /** $OpDocMathSegmentProd
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentProd[I <: Int32OrInt64](segmentIndices: Tensor[I]): Tensor[D] = Math.segmentProd(tensor, segmentIndices)

      /** $OpDocMathSegmentMin
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentMin[I <: Int32OrInt64](segmentIndices: Tensor[I]): Tensor[D] = Math.segmentMin(tensor, segmentIndices)

      /** $OpDocMathSegmentMax
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentMax[I <: Int32OrInt64](segmentIndices: Tensor[I]): Tensor[D] = Math.segmentMax(tensor, segmentIndices)

      //endregion Segment Ops

      //region Other Ops

      /** $OpDocMathZerosFraction
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def zerosFraction: Tensor[FLOAT32] = Math.zerosFraction(tensor)

      //endregion Other Ops
    }

    implicit class MathMathOps[D <: MathDataType](val tensor: Tensor[D]) {
      //region Operators

      /** $OpDocMathNegate
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def unary_- : Tensor[D] = negate

      /** $OpDocMathAdd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def +(other: Tensor[D]): Tensor[D] = add(other)

      /** $OpDocMathSubtract
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def -(other: Tensor[D]): Tensor[D] = subtract(other)

      /** $OpDocMathMultiply
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def *(other: Tensor[D]): Tensor[D] = multiply(other)

      private[this] def divHelper(x: Tensor[D], y: Tensor[D]): Tensor[D] = {
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
      def /(other: Tensor[D]): Tensor[D] = divHelper(tensor, other)

      /** $OpDocMathMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def %(other: Tensor[D]): Tensor[D] = mod(other)

      /** $OpDocMathPow
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def **(other: Tensor[D]): Tensor[D] = pow(other)

      /** $OpDocMathPow
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ^(other: Tensor[D]): Tensor[D] = pow(other)

      //endregion Operators

      //region Unary Ops

      /** $OpDocMathAbs
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def abs: Tensor[D] = Math.abs(tensor)

      /** $OpDocMathNegate
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def negate: Tensor[D] = Math.negate(tensor)

      /** $OpDocMathReciprocal
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def reciprocal: Tensor[D] = Math.reciprocal(tensor)

      /** $OpDocMathSquare
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def square: Tensor[D] = Math.square(tensor)

      /** $OpDocMathSqrt
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sqrt: Tensor[D] = Math.sqrt(tensor)

      /** $OpDocMathRsqrt
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def rsqrt: Tensor[D] = Math.rsqrt(tensor)

      /** $OpDocMathExp
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def exp: Tensor[D] = Math.exp(tensor)

      /** $OpDocMathExpm1
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def expm1: Tensor[D] = Math.expm1(tensor)

      /** $OpDocMathLog
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def log: Tensor[D] = Math.log(tensor)

      /** $OpDocMathLog1p
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def log1p: Tensor[D] = Math.log1p(tensor)

      /** $OpDocMathSin
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sin: Tensor[D] = Math.sin(tensor)

      /** $OpDocMathCos
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def cos: Tensor[D] = Math.cos(tensor)

      /** $OpDocMathTan
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def tan: Tensor[D] = Math.tan(tensor)

      /** $OpDocMathAsin
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def asin: Tensor[D] = Math.asin(tensor)

      /** $OpDocMathAcos
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def acos: Tensor[D] = Math.acos(tensor)

      /** $OpDocMathAtan
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def atan: Tensor[D] = Math.atan(tensor)

      /** $OpDocMathSinh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sinh: Tensor[D] = Math.sinh(tensor)

      /** $OpDocMathCosh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def cosh: Tensor[D] = Math.cosh(tensor)

      /** $OpDocMathTanh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def tanh: Tensor[D] = Math.tanh(tensor)

      /** $OpDocMathAsinh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def asinh: Tensor[D] = Math.asinh(tensor)

      /** $OpDocMathAcosh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def acosh: Tensor[D] = Math.acosh(tensor)

      /** $OpDocMathAtanh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def atanh: Tensor[D] = Math.atanh(tensor)

      /** $OpDocMathLogGamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logGamma: Tensor[D] = Math.logGamma(tensor)

      /** $OpDocMathDigamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def digamma: Tensor[D] = Math.digamma(tensor)

      /** $OpDocMathErf
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def erf: Tensor[D] = Math.erf(tensor)

      /** $OpDocMathErfc
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def erc: Tensor[D] = Math.erfc(tensor)

      /** $OpDocMathSigmoid
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sigmoid: Tensor[D] = Math.sigmoid(tensor)

      /** $OpDocMathSign
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sign: Tensor[D] = Math.sign(tensor)

      /** $OpDocMathRound
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def round: Tensor[D] = Math.round(tensor)

      //endregion Unary Ops

      //region Binary Ops

      /** $OpDocMathAdd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def add(other: Tensor[D]): Tensor[D] = Math.add(tensor, other)

      /** $OpDocMathSubtract
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def subtract(other: Tensor[D]): Tensor[D] = Math.subtract(tensor, other)

      /** $OpDocMathMultiply
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def multiply(other: Tensor[D]): Tensor[D] = Math.multiply(tensor, other)

      /** $OpDocMathDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def divide(other: Tensor[D]): Tensor[D] = Math.divide(tensor, other)

      /** $OpDocMathFloorDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      @deprecated("Use `truncateDivide` instead.", "0.1")
      def floorDivide(other: Tensor[D]): Tensor[D] = Math.floorDivide(tensor, other)

      /** $OpDocMathTruncateDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def truncateDivide(other: Tensor[D]): Tensor[D] = Math.truncateDivide(tensor, other)

      /** $OpDocMathRealDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def realDivide(other: Tensor[D]): Tensor[D] = Math.realDivide(tensor, other)

      /** $OpDocMathSquaredDifference
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def squaredDifference(other: Tensor[D]): Tensor[D] = Math.squaredDifference(tensor, other)

      /** $OpDocMathMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def mod(other: Tensor[D]): Tensor[D] = Math.mod(tensor, other)

      /** $OpDocMathFloorMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def floorMod(other: Tensor[D]): Tensor[D] = Math.floorMod(tensor, other)

      /** $OpDocMathTruncateMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def truncateMod(other: Tensor[D]): Tensor[D] = Math.truncateMod(tensor, other)

      /** $OpDocMathPow
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def pow(other: Tensor[D]): Tensor[D] = Math.pow(tensor, other)

      /** $OpDocMathMaximum
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def maximum(other: Tensor[D]): Tensor[D] = Math.maximum(tensor, other)

      /** $OpDocMathMinimum
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def minimum(other: Tensor[D]): Tensor[D] = Math.minimum(tensor, other)

      //endregion Binary Ops

      //region Reduction Ops

      /** $OpDocMathLogSumExp
        *
        * @group MathOps
        * @param  axes     Integer sequence containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def logSumExp(axes: Seq[Int] = null, keepDims: Boolean = false): Tensor[D] = Math.logSumExp(tensor, axes, keepDims)

      //endregion Reduction Ops

      /** $OpDocMathArgmax
        *
        * @group MathOps
        * @param  axes Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @return Result as a new tensor.
        */
      def argmax[I <: Int32OrInt64](axes: Tensor[I]): Tensor[INT64] = Math.argmax(tensor, axes)

      /** $OpDocMathArgmax
        *
        * @group MathOps
        * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  outputDataType Data type for the output tensor.
        * @return Result as a new tensor.
        */
      def argmax[I <: Int32OrInt64, IR <: Int32OrInt64](
          axes: Tensor[I],
          outputDataType: IR
      ): Tensor[IR] = {
        Math.argmax(tensor, axes, outputDataType)
      }

      /** $OpDocMathArgmin
        *
        * @group MathOps
        * @param  axes Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @return Result as a new tensor.
        */
      def argmin[I <: Int32OrInt64](axes: Tensor[I]): Tensor[INT64] = Math.argmin(tensor, axes)

      /** $OpDocMathArgmin
        *
        * @group MathOps
        * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  outputDataType Data type for the output tensor.
        * @return Result as a new tensor.
        */
      def argmin[I <: Int32OrInt64, IR <: Int32OrInt64](
          axes: Tensor[I],
          outputDataType: IR
      ): Tensor[IR] = {
        Math.argmin(tensor, axes, outputDataType)
      }

      /** $OpDocMathCumsum
        *
        * @group MathOps
        * @param  axis      Tensor containing the axis along which to perform the cumulative sum.
        * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
        * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
        * @return Result as a new tensor.
        */
      def cumsum(axis: Tensor[INT32] = 0, exclusive: Boolean = false, reverse: Boolean = false): Tensor[D] = {
        Math.cumsum(tensor, axis, exclusive, reverse)
      }

      /** $OpDocMathCumprod
        *
        * @group MathOps
        * @param  axis      Tensor containing the axis along which to perform the cumulative product.
        * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative product.
        * @param  reverse   Boolean value indicating whether to perform a reverse cumulative product.
        * @return Result as a new tensor.
        */
      def cumprod(axis: Tensor[INT32] = 0, exclusive: Boolean = false, reverse: Boolean = false): Tensor[D] = {
        Math.cumprod(tensor, axis, exclusive, reverse)
      }

      //region Matrix Ops

      /** $OpDocMathDiag
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def diag: Tensor[D] = Math.diag(tensor)

      /** $OpDocMathDiagPart
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def diagPart: Tensor[D] = Math.diagPart(tensor)

      /** $OpDocMathMatrixDiag
        *
        * @group MathOps
        * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its
        *         last dimension duplicated.
        */
      def matrixDiag: Tensor[D] = Math.matrixDiag(tensor)

      /** $OpDocMathMatrixSetDiag
        *
        * @group MathOps
        * @param  diagonal Rank-`K` tensor, where `K >= 1`.
        * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `input`.
        */
      def matrixSetDiag(diagonal: Tensor[D]): Tensor[D] = Math.matrixSetDiag(tensor, diagonal)

      /** $OpDocMathMatrixDiagPart
        *
        * @group MathOps
        * @return Result as a new tensor containing the diagonal(s) and having shape equal to
        *         `input.shape[:-2] + [min(input.shape[-2:])]`.
        */
      def matrixDiagPart: Tensor[D] = Math.matrixDiagPart(tensor)

      /** $OpDocMathMatrixBandPart
        *
        * @group MathOps
        * @param  numSubDiagonals   Scalar tensor that contains the number of sub-diagonals to keep. If negative,
        *                           the entire lower triangle is kept.
        * @param  numSuperDiagonals Scalar tensor that contains the number of super-diagonals to keep. If negative,
        *                           the entire upper triangle is kept.
        * @return Result as a new tensor containing the expected banded tensor and has rank `K` and same shape as `input`.
        */
      def matrixBandPart[I <: Int32OrInt64](numSubDiagonals: Tensor[I], numSuperDiagonals: Tensor[I]): Tensor[D] = {
        Math.matrixBandPart(tensor, numSubDiagonals, numSuperDiagonals)
      }

      /** $OpDocMathTrace
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def trace: Tensor[D] = Math.trace(tensor)

      /** $OpDocMathMatmul
        *
        * @group MathOps
        * @param  other      Tensor to multiply with.
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
          other: Tensor[D],
          transposeA: Boolean = false,
          transposeB: Boolean = false,
          conjugateA: Boolean = false,
          conjugateB: Boolean = false,
          aIsSparse: Boolean = false,
          bIsSparse: Boolean = false
      ): Tensor[D] = {
        Math.matmul(tensor, other, transposeA, transposeB, conjugateA, conjugateB, aIsSparse, bIsSparse)
      }

      /** $OpDocMathCross
        *
        * @group MathOps
        * @param  other Tensor to multiply with.
        * @return Result as a new tensor.
        */
      def cross(other: Tensor[D]): Tensor[D] = Math.cross(tensor, other)

      /** Dynamic version (i.e., where `numAxes` may be a tensor) of the `tensorDot` op.
        *
        * $OpDocMathTensorDot
        *
        * @group MathOps
        * @param  other   Tensor to contract with.
        * @param  numAxes Number of axes to contract.
        * @return Created op output.
        */
      def tensorDot(other: Tensor[D], numAxes: Tensor[INT32]): Tensor[D] = {
        Math.tensorDot(tensor, other, numAxes)
      }

      /** Dynamic version (i.e., where `axesA` and `axesB` may be tensors) of the `tensorDot` op.
        *
        * $OpDocMathTensorDot
        *
        * @group MathOps
        * @param  other Tensor to contract with.
        * @param  axesA Axes to contract in `a`.
        * @param  axesB Axes to contract in `b`.
        * @return Created op output.
        */
      def tensorDot(other: Tensor[D], axesA: Tensor[INT32], axesB: Tensor[INT32]): Tensor[D] = {
        Math.tensorDot(tensor, other, axesA, axesB)
      }

      //endregion Matrix Ops
    }

    implicit class Int32OrInt64OrFloat32OrFloat64MathOps[D <: Int32OrInt64OrFloat32OrFloat64](val tensor: Tensor[D]) {
      //region Bucketization Ops

      /** $OpDocMathBucketize
        *
        * @group MathOps
        * @param  boundaries Sorted sequence of numbers specifying the boundaries of the buckets.
        * @return Result as a new tensor.
        */
      def bucketize(boundaries: Seq[Float]): Tensor[D] = Math.bucketize(tensor, boundaries)

      //endregion Bucketization Ops
    }

    implicit class Float16OrFloat32OrFloat64MathOps[D <: Float16OrFloat32OrFloat64](val tensor: Tensor[D]) {
      //region Unary Ops

      /** $OpDocMathRoundInt
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def roundInt: Tensor[D] = Math.roundInt(tensor)

      /** $OpDocMathFloor
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def floor: Tensor[D] = Math.floor(tensor)

      /** $OpDocMathCeil
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ceil: Tensor[D] = Math.ceil(tensor)

      /** $OpDocMathIsNaN
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def isNaN: Tensor[BOOLEAN] = Math.isNaN(tensor)

      /** $OpDocMathIsInf
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def isInf: Tensor[BOOLEAN] = Math.isInf(tensor)

      /** $OpDocMathIsFinite
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def isFinite: Tensor[BOOLEAN] = Math.isFinite(tensor)

      //endregion Unary Ops
    }

    implicit class Float32OrFloat64MathOps[D <: Float32OrFloat64](val tensor: Tensor[D]) {
      //region Binary Ops

      /** $OpDocMathIgammac
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def igammac(other: Tensor[D]): Tensor[D] = Math.igammac(tensor, other)

      /** $OpDocMathIgamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def igamma(other: Tensor[D]): Tensor[D] = Math.igamma(tensor, other)

      /** $OpDocMathZeta
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def zeta(other: Tensor[D]): Tensor[D] = Math.zeta(tensor, other)

      /** $OpDocMathPolygamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def polygamma(other: Tensor[D]): Tensor[D] = Math.polygamma(tensor, other)

      /** $OpDocMathAtan2
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def atan2(other: Tensor[D]): Tensor[D] = Math.atan2(tensor, other)

      //endregion Binary Ops
    }

    implicit class ComplexMathOps[D <: ComplexDataType](val tensor: Tensor[D]) {
      /** $OpDocMathConjugate
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def conjugate: Tensor[D] = Math.conjugate(tensor)
    }

    implicit class BooleanMathOps(val tensor: Tensor[BOOLEAN]) {
      //region Operators

      /** $OpDocMathLogicalNot
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def unary_! : Tensor[BOOLEAN] = logicalNot

      /** $OpDocMathLogicalAnd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def &&(other: Tensor[BOOLEAN]): Tensor[BOOLEAN] = logicalAnd(other)

      /** $OpDocMathLogicalOr
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ||(other: Tensor[BOOLEAN]): Tensor[BOOLEAN] = logicalOr(other)

      //endregion Operators

      //region Logical Ops

      /** $OpDocMathLogicalNot
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalNot: Tensor[BOOLEAN] = Math.logicalNot(tensor)

      /** $OpDocMathLogicalAnd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalAnd(other: Tensor[BOOLEAN]): Tensor[BOOLEAN] = Math.logicalAnd(tensor, other)

      /** $OpDocMathLogicalOr
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalOr(other: Tensor[BOOLEAN]): Tensor[BOOLEAN] = Math.logicalOr(tensor, other)

      /** $OpDocMathLogicalXOr
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalXOr(other: Tensor[BOOLEAN]): Tensor[BOOLEAN] = Math.logicalXOr(tensor, other)

      //endregion Logical Ops

      //region Reduction Ops

      /** $OpDocMathAll
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def all(axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[BOOLEAN] = Math.all(tensor, axes, keepDims)

      /** $OpDocMathAny
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def any(axes: Tensor[INT32] = null, keepDims: Boolean = false): Tensor[BOOLEAN] = Math.any(tensor, axes, keepDims)

      //endregion Reduction Ops
    }

    implicit class Int32MathOps(val tensor: Tensor[INT32]) {
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
      def binCount[D <: Int32OrInt64OrFloat32OrFloat64](
          weights: Tensor[D] = null,
          minLength: Tensor[INT32] = null,
          maxLength: Tensor[INT32] = null,
          dataType: D = null
      ): Tensor[D] = {
        Math.binCount(tensor, weights, minLength, maxLength, dataType)
      }
    }

    implicit class Complex64MathOps(val tensor: Tensor[COMPLEX64]) {
      /** $OpDocMathReal
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def real: Tensor[FLOAT32] = Math.real(tensor)

      /** $OpDocMathImag
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def imag: Tensor[FLOAT32] = Math.imag(tensor)

      /** $OpDocMathAngle
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def angle: Tensor[FLOAT32] = Math.angle(tensor)
    }

    implicit class Complex128MathOps(val tensor: Tensor[COMPLEX128]) {
      /** $OpDocMathReal
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def real: Tensor[FLOAT64] = Math.real(tensor)

      /** $OpDocMathImag
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def imag: Tensor[FLOAT64] = Math.imag(tensor)

      /** $OpDocMathAngle
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def angle: Tensor[FLOAT64] = Math.angle(tensor)
    }

    implicit def tensorConvertibleToMathOps[D <: DataType, T](value: T)(implicit f: T => Tensor[D]): MathOps[D] = new MathOps(f(value))
    implicit def tensorConvertibleToReducibleMathOps[D <: ReducibleDataType, T](value: T)(implicit f: T => Tensor[D]): ReducibleMathOps[D] = new ReducibleMathOps(f(value))
    implicit def tensorConvertibleToMathMathOps[D <: MathDataType, T](value: T)(implicit f: T => Tensor[D]): MathMathOps[D] = new MathMathOps(f(value))
    implicit def tensorConvertibleToInt32OrInt64OrFloat32OrFloat64MathOps[D <: Int32OrInt64OrFloat32OrFloat64, T](value: T)(implicit f: T => Tensor[D]): Int32OrInt64OrFloat32OrFloat64MathOps[D] = new Int32OrInt64OrFloat32OrFloat64MathOps(f(value))
    implicit def tensorConvertibleToFloat16OrFloat32OrFloat64MathOps[D <: Float16OrFloat32OrFloat64, T](value: T)(implicit f: T => Tensor[D]): Float16OrFloat32OrFloat64MathOps[D] = new Float16OrFloat32OrFloat64MathOps(f(value))
    implicit def tensorConvertibleToFloat32OrFloat64MathOps[D <: Float32OrFloat64, T](value: T)(implicit f: T => Tensor[D]): Float32OrFloat64MathOps[D] = new Float32OrFloat64MathOps(f(value))
    implicit def tensorConvertibleToComplexMathOps[D <: ComplexDataType, T](value: T)(implicit f: T => Tensor[D]): ComplexMathOps[D] = new ComplexMathOps(f(value))
    implicit def tensorConvertibleToBooleanMathOps[T](value: T)(implicit f: T => Tensor[BOOLEAN]): BooleanMathOps = new BooleanMathOps(f(value))
    implicit def tensorConvertibleToInt32MathOps[T](value: T)(implicit f: T => Tensor[INT32]): Int32MathOps = new Int32MathOps(f(value))
    implicit def tensorConvertibleToComplex64MathOps[T](value: T)(implicit f: T => Tensor[COMPLEX64]): Complex64MathOps = new Complex64MathOps(f(value))
    implicit def tensorConvertibleToComplex128MathOps[T](value: T)(implicit f: T => Tensor[COMPLEX128]): Complex128MathOps = new Complex128MathOps(f(value))
  }
}
