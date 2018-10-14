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
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault
import org.platanios.tensorflow.jni.generated.tensors.{Math => NativeTensorOpsMath}

/** Contains functions for executing general math-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Math {
  /** $OpDocMathSelect
    *
    * @group MathOps
    * @param  condition Condition tensor.
    * @param  x         Tensor which may have the same shape as `condition`. If `condition` has rank `1`, then `t` may
    *                   have a higher rank, but its first dimension must match the size of `condition`.
    * @param  y         Tensor with the same data type and shape as `t`.
    * @return Result as a new tensor.
    */
  def select[T: TF](
      condition: Tensor[Boolean],
      x: Tensor[T],
      y: Tensor[T]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.select(
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
  def range[T: TF : IsNumeric](
      start: Tensor[T],
      limit: Tensor[T],
      delta: Tensor[T] = null
  ): Tensor[T] = {
    val deltaWithDefault = if (delta == null) Tensor.ones(start.dataType, Shape()) else delta
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.range(
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
  def linspace[T: TF : IsBFloat16OrFloat32OrFloat64, I: TF : IsInt32OrInt64](
      start: Tensor[T],
      stop: Tensor[T],
      numberOfValues: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.linSpace(
      executionContext.value.nativeHandle, start.nativeHandle, stop.nativeHandle, numberOfValues.nativeHandle))
  }

  /** $OpDocMathAddN
    *
    * @group MathOps
    * @param  inputs Input tensors.
    * @return Result as a new tensor.
    */
  def addN[T: TF : IsNumeric](inputs: Seq[Tensor[T]]): Tensor[T] = {
    if (inputs.length == 1)
      inputs.head
    else
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.addN(
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
  def abs[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    if (x.dataType.isComplex) {
      ev.applyUnary(x, t => {
        Tensor.fromNativeHandle[T](NativeTensorOpsMath.complexAbs(
          executionContext.value.nativeHandle, t.nativeHandle, x.dataType.cValue))
      })
    } else {
      ev.applyUnary(x, t => {
        Tensor.fromNativeHandle[T](NativeTensorOpsMath.abs(executionContext.value.nativeHandle, t.nativeHandle))
      })
    }
  }

  /** $OpDocMathNegate
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def negate[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.neg(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathReciprocal
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def reciprocal[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.reciprocal(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSquare
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def square[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.square(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSqrt
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sqrt[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sqrt(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathRsqrt
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def rsqrt[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.rsqrt(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathExp
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def exp[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.exp(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathExpm1
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def expm1[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.expm1(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathLog
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def log[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.log(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathLog1p
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def log1p[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.log1p(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSin
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sin[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sin(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathCos
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def cos[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.cos(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathTan
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def tan[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.tan(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAsin
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def asin[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.asin(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAcos
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def acos[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.acos(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAtan
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def atan[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.atan(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSinh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sinh[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sinh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathCosh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def cosh[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.cosh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathTanh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def tanh[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.tanh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAsinh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def asinh[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.asinh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAcosh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def acosh[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.acosh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathAtanh
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def atanh[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.atanh(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathLogGamma
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def logGamma[T: TF : IsFloat32OrFloat64, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.lgamma(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathDigamma
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def digamma[T: TF : IsFloat32OrFloat64, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.digamma(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathErf
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def erf[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.erf(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathErfc
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def erfc[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.erfc(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathSigmoid
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sigmoid[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sigmoid(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathLogSigmoid
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def logSigmoid[T: TF : IsReal, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    negate(NN.softplus(negate(x)))
  }

  /** $OpDocMathSign
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def sign[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sign(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathRound
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def round[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](x: TL[T])(implicit
      ev: TensorOps.Aux[TL, T]
  ): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.round(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathRoundInt
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def roundInt[T: TF : IsFloat16OrFloat32OrFloat64, TL[A] <: TensorLike[A]](
      x: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.rint(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathFloor
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def floor[T: TF : IsFloat16OrFloat32OrFloat64, TL[A] <: TensorLike[A]](
      x: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.floor(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathCeil
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def ceil[T: TF : IsFloat16OrFloat32OrFloat64, TL[A] <: TensorLike[A]](
      x: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[T] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.ceil(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathIsNaN
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def isNaN[T: TF : IsFloat16OrFloat32OrFloat64, TL[A] <: TensorLike[A]](
      x: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[Boolean] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.isNan(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathIsInf
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def isInf[T: TF : IsFloat16OrFloat32OrFloat64, TL[A] <: TensorLike[A]](
      x: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[Boolean] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.isInf(executionContext.value.nativeHandle, t.nativeHandle))
    })
  }

  /** $OpDocMathIsFinite
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def isFinite[T: TF : IsFloat16OrFloat32OrFloat64, TL[A] <: TensorLike[A]](
      x: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[Boolean] = {
    ev.applyUnary(x, t => {
      Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.isFinite(
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
  def add[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.add(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathSubtract
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def subtract[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.sub(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathMultiply
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def multiply[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.mul(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathDivide
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def divide[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.div(
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
  def floorDivide[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.floorDiv(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathTruncateDivide
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def truncateDivide[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.truncateDiv(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathRealDivide
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def realDivide[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.realDiv(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathSquaredDifference
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def squaredDifference[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.squaredDifference(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathMod
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def mod[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.mod(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathFloorMod
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def floorMod[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.floorMod(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathTruncateMod
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def truncateMod[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.truncateMod(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathPow
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def pow[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.pow(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  // TODO: !!! [TENSORS] Fix this.
  
  /** $OpDocMathIgammac
    *
    * @group MathOps
    * @param  a First input tensor.
    * @param  x Second input tensor.
    * @return Result as a new tensor.
    */
  def igammac[T: TF : IsFloat32OrFloat64](a: Tensor[T], x: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.igammac(
      executionContext.value.nativeHandle, a.nativeHandle, x.nativeHandle))
  }

  /** $OpDocMathIgamma
    *
    * @group MathOps
    * @param  a First input tensor.
    * @param  x Second input tensor.
    * @return Result as a new tensor.
    */
  def igamma[T: TF : IsFloat32OrFloat64](a: Tensor[T], x: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.igamma(
      executionContext.value.nativeHandle, a.nativeHandle, x.nativeHandle))
  }

  /** $OpDocMathZeta
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  q Second input tensor.
    * @return Result as a new tensor.
    */
  def zeta[T: TF : IsFloat32OrFloat64](x: Tensor[T], q: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.zeta(
      executionContext.value.nativeHandle, x.nativeHandle, q.nativeHandle))
  }

  /** $OpDocMathPolygamma
    *
    * @group MathOps
    * @param  n First input tensor.
    * @param  x Second input tensor.
    * @return Result as a new tensor.
    */
  def polygamma[T: TF : IsFloat32OrFloat64](n: Tensor[T], x: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.polygamma(
      executionContext.value.nativeHandle, n.nativeHandle, x.nativeHandle))
  }

  /** $OpDocMathAtan2
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def atan2[T: TF : IsFloat32OrFloat64](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.atan2(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathMinimum
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def minimum[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.minimum(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathMaximum
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def maximum[T: TF : IsNotQuantized](x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.maximum(
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
  def incompleteBeta[T: TF : IsFloat32OrFloat64](a: Tensor[T], b: Tensor[T], x: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.betainc(
      executionContext.value.nativeHandle, a.nativeHandle, b.nativeHandle, x.nativeHandle))
  }

  //region Logical Ops

  /** $OpDocMathLogicalNot
    *
    * @group MathOps
    * @param  x Input tensor.
    * @return Result as a new tensor.
    */
  def logicalNot(x: Tensor[Boolean]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.logicalNot(
      executionContext.value.nativeHandle, x.nativeHandle))
  }

  /** $OpDocMathLogicalAnd
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def logicalAnd(x: Tensor[Boolean], y: Tensor[Boolean]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.logicalAnd(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathLogicalOr
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def logicalOr(x: Tensor[Boolean], y: Tensor[Boolean]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.logicalOr(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathLogicalXOr
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def logicalXOr(x: Tensor[Boolean], y: Tensor[Boolean]): Tensor[Boolean] = {
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
  def equal[T: TF : IsNumeric](x: Tensor[T], y: Tensor[T]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.equal(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** $OpDocMathNotEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def notEqual[T: TF : IsNumeric](x: Tensor[T], y: Tensor[T]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.notEqual(
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
  def approximatelyEqual[T: TF : IsNumeric](
      x: Tensor[T],
      y: Tensor[T],
      tolerance: Float = 0.00001f
  ): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.approximateEqual(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, tolerance))
  }

  /** OpDocMathLess
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def less[T: TF : IsNumeric](x: Tensor[T], y: Tensor[T]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.less(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** OpDocMathLessEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def lessEqual[T: TF : IsNumeric](x: Tensor[T], y: Tensor[T]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.lessEqual(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** OpDocMathGreater
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def greater[T: TF : IsNumeric](x: Tensor[T], y: Tensor[T]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.greater(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  /** OpDocMathGreaterEqual
    *
    * @group MathOps
    * @param  x First input tensor.
    * @param  y Second input tensor.
    * @return Result as a new tensor.
    */
  def greaterEqual[T: TF : IsNumeric](x: Tensor[T], y: Tensor[T]): Tensor[Boolean] = {
    Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.greaterEqual(
      executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  //endregion Comparison Ops

  //region Reduction Ops

  private def defaultReductionAxes[T: TF, TL[A] <: TensorLike[A]](
      tensorLike: TL[T]
  ): Tensor[Int] = {
    // Fast path: Avoid creating range and rank ops if the rank is known statically.
    val reductionAxes = tensorLike match {
      case t: Tensor[T] if t.rank > -1 => (0 until t.rank).toTensor
      case t: TensorIndexedSlices[T] if t.denseShape.shape.isFullyDefined =>
        (0 until t.denseShape.shape(0)).toTensor
      case t: SparseTensor[T] if t.denseShape.shape.isFullyDefined =>
        (0 until t.denseShape.shape(0)).toTensor
      case _ => // Otherwise, we rely on range and rank to do the right thing at run-time.
        range(0, Basic.rank(tensorLike))
    }
    reductionAxes
  }

  /** $OpDocMathSum
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def sum[T: TF : IsNumeric, I: IntDefault : TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I] = null,
      keepDims: Boolean = false
  ): Tensor[T] = {
    if (input.rank == 0) {
      input
    } else if (axes == null) {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sum(
        executionContext.value.nativeHandle, input.nativeHandle, defaultReductionAxes(input).nativeHandle, keepDims))
    } else {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sum(
        executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, keepDims))
    }
  }

  /** $OpDocMathMean
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def mean[T: TF : IsNumeric, I: IntDefault : TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I] = null,
      keepDims: Boolean = false
  ): Tensor[T] = {
    if (input.rank == 0) {
      input
    } else if (axes == null) {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.mean(
        executionContext.value.nativeHandle, input.nativeHandle, defaultReductionAxes(input).nativeHandle, keepDims))
    } else {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.mean(
        executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, keepDims))
    }
  }

  /** $OpDocMathProd
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def prod[T: TF : IsNotQuantized, I: IntDefault : TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I] = null,
      keepDims: Boolean = false
  ): Tensor[T] = {
    if (input.rank == 0) {
      input
    } else if (axes == null) {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.prod(
        executionContext.value.nativeHandle, input.nativeHandle, defaultReductionAxes(input).nativeHandle, keepDims))
    } else {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.prod(
        executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, keepDims))
    }
  }

  /** $OpDocMathMin
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def min[T: TF : IsNotQuantized, I: IntDefault : TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I] = null,
      keepDims: Boolean = false
  ): Tensor[T] = {
    if (input.rank == 0) {
      input
    } else if (axes == null) {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.min(
        executionContext.value.nativeHandle, input.nativeHandle, defaultReductionAxes(input).nativeHandle, keepDims))
    } else {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.min(
        executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, keepDims))
    }
  }

  /** $OpDocMathMax
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def max[T: TF : IsNotQuantized, I: IntDefault : TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I] = null,
      keepDims: Boolean = false
  ): Tensor[T] = {
    if (input.rank == 0) {
      input
    } else if (axes == null) {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.max(
        executionContext.value.nativeHandle, input.nativeHandle, defaultReductionAxes(input).nativeHandle, keepDims))
    } else {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.max(
        executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, keepDims))
    }
  }

  /** $OpDocMathAll
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def all[I: IntDefault : TF : IsInt32OrInt64](
      input: Tensor[Boolean],
      axes: Tensor[I] = null,
      keepDims: Boolean = false
  ): Tensor[Boolean] = {
    if (input.rank == 0) {
      input
    } else if (axes == null) {
      Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.all(
        executionContext.value.nativeHandle, input.nativeHandle, defaultReductionAxes(input).nativeHandle, keepDims))
    } else {
      Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.all(
        executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, keepDims))
    }
  }

  /** $OpDocMathAny
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def any[I: IntDefault : TF : IsInt32OrInt64](
      input: Tensor[Boolean],
      axes: Tensor[I] = null,
      keepDims: Boolean = false
  ): Tensor[Boolean] = {
    if (input.rank == 0) {
      input
    } else if (axes == null) {
      Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.any(
        executionContext.value.nativeHandle, input.nativeHandle, defaultReductionAxes(input).nativeHandle, keepDims))
    } else {
      Tensor.fromNativeHandle[Boolean](NativeTensorOpsMath.any(
        executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, keepDims))
    }
  }

  /** $OpDocMathLogSumExp
    *
    * @group MathOps
    * @param  input    Input tensor to reduce.
    * @param  axes     Integer sequence containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def logSumExp[T: TF : IsNotQuantized](
      input: Tensor[T],
      axes: Seq[Int] = null,
      keepDims: Boolean = false
  ): Tensor[T] = {
    if (input.rank == 0) {
      input
    } else {
      val axesTensor: Tensor[Int] = axes
      val maxValue = Basic.stopGradient(max(input, axesTensor, keepDims = true))
      val result = add(log(sum(exp(subtract(input, maxValue)), axesTensor, keepDims = true)), maxValue)
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
    * @param  axes     Tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @param  keepDims If `true`, retain the reduced axes.
    * @return Result as a new tensor.
    */
  def countNonZero[T: TF : IsNumeric, I: IntDefault : TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I] = null,
      keepDims: Boolean = false
  ): Tensor[Long] = {
    sum(notEqual(input, Tensor.zeros(input.dataType, Shape())).castTo[Long], axes, keepDims)
  }

  /** $OpDocMathArgmin
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  axes  Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @return Result as a new tensor.
    */
  def argmin[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I]
  ): Tensor[Long] = {
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
  def argmin[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64, IR: TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I],
      outputDataType: DataType[IR]
  ): Tensor[IR] = {
    Tensor.fromNativeHandle[IR](NativeTensorOpsMath.argMin(
      executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, outputDataType.cValue))
  }

  /** $OpDocMathArgmax
    *
    * @group MathOps
    * @param  input Input tensor.
    * @param  axes  Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
    * @return Result as a new tensor.
    */
  def argmax[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I]
  ): Tensor[Long] = {
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
  def argmax[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64, IR: TF : IsInt32OrInt64](
      input: Tensor[T],
      axes: Tensor[I],
      outputDataType: DataType[IR]
  ): Tensor[IR] = {
    Tensor.fromNativeHandle[IR](NativeTensorOpsMath.argMax(
      executionContext.value.nativeHandle, input.nativeHandle, axes.nativeHandle, outputDataType.cValue))
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
  def cumsum[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      axis: Tensor[I],
      exclusive: Boolean = false,
      reverse: Boolean = false
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.cumsum(
      executionContext.value.nativeHandle, input.nativeHandle, axis.nativeHandle, exclusive, reverse))
  }

  /** $OpDocMathCumprod
    *
    * @group MathOps
    * @param  input     Input tensor.
    * @param  axis      Tensor containing the axis along which to perform the cumulative product.
    * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative product.
    * @param  reverse   Boolean value indicating whether to perform a reverse cumulative product.
    * @return Result as a new tensor.
    */
  def cumprod[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      axis: Tensor[I],
      exclusive: Boolean = false,
      reverse: Boolean = false
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.cumprod(
      executionContext.value.nativeHandle, input.nativeHandle, axis.nativeHandle, exclusive, reverse))
  }

  //endregion Reduction Ops

  /** $OpDocMathBinCount
    *
    * @group MathOps
    * @param  input     Tensor containing non-negative values.
    * @param  dataType  If `weights` is `null`, this determines the data type used for the output tensor (i.e., the
    *                   tensor containing the bin counts).
    * @param  weights   If not `null`, this tensor must have the same shape as `input`. For each value in `input`, the
    *                   corresponding bin count will be incremented by the corresponding weight instead of `1`.
    * @param  minLength If not `null`, this ensures the output has length at least `minLength`, padding with zeros at
    *                   the end, if necessary.
    * @param  maxLength If not `null`, this skips values in `input` that are equal or greater than `maxLength`,
    *                   ensuring that the output has length at most `maxLength`.
    * @return Result as a new tensor.
    */
  def binCount[T: TF : IsInt32OrInt64OrFloat32OrFloat64](
      input: Tensor[Int],
      dataType: DataType[T],
      weights: Tensor[T] = null,
      minLength: Tensor[Int] = null,
      maxLength: Tensor[Int] = null
  ): Tensor[T] = {
    val inputNonEmpty = greater(prod(Basic.shape(input).toInt), 0)
    var outputSize = inputNonEmpty.toInt * add(max(input), Tensor.ones[Int](Shape()))
    if (minLength != null)
      outputSize = maximum(minLength, outputSize)
    if (maxLength != null)
      outputSize = minimum(maxLength, outputSize)
    val effectiveWeights = {
      if (weights != null) {
        weights
      } else if (dataType == null) {
        Tensor.zeros[Int](Shape.scalar())
      } else {
        Tensor.zeros(dataType, Shape.scalar())
      }
    }
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.bincount(
      executionContext.value.nativeHandle, input.nativeHandle, outputSize.nativeHandle, effectiveWeights.nativeHandle))
  }

  //region Segment Ops

  /** $OpDocMathSegmentSum
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentSum[T: TF : IsNumeric, I: TF : IsInt32OrInt64](
      data: Tensor[T],
      segmentIndices: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.segmentSum(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentMean
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentMean[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      data: Tensor[T],
      segmentIndices: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.segmentMean(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentProd
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentProd[T: TF : IsNumeric, I: TF : IsInt32OrInt64](
      data: Tensor[T],
      segmentIndices: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.segmentProd(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentMin
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentMin[T: TF : IsReal, I: TF : IsInt32OrInt64](
      data: Tensor[T],
      segmentIndices: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.segmentMin(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathSegmentMax
    *
    * @group MathOps
    * @param  data           Data (must have a numeric data type -- i.e., representing a number).
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @return Result as a new tensor.
    */
  def segmentMax[T: TF : IsReal, I: TF : IsInt32OrInt64](
      data: Tensor[T],
      segmentIndices: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.segmentMax(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle))
  }

  /** $OpDocMathUnsortedSegmentSum
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @return Result as a new tensor.
    */
  def unsortedSegmentSum[T: TF : IsNumeric, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      data: Tensor[T],
      segmentIndices: Tensor[I1],
      segmentsNumber: Tensor[I2]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.unsortedSegmentSum(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle, segmentsNumber.nativeHandle))
  }

  // TODO: [TENSORS] Missing 'unsortedSegmentMean'.
  // TODO: [TENSORS] Missing 'unsortedSegmentProd'.
  // TODO: [TENSORS] Missing 'unsortedSegmentMin'.

  /** $OpDocMathUnsortedSegmentMax
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  segmentIndices Segment indices.
    * @param  segmentsNumber Number of segments.
    * @return Result as a new tensor.
    */
  def unsortedSegmentMax[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
      data: Tensor[T],
      segmentIndices: Tensor[I1],
      segmentsNumber: Tensor[I2]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.unsortedSegmentMax(
      executionContext.value.nativeHandle, data.nativeHandle, segmentIndices.nativeHandle, segmentsNumber.nativeHandle))
  }

  // TODO: [TENSORS] Missing 'unsortedSegmentSqrtN'.

  /** $OpDocMathSparseSegmentSum
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  numSegments    Optional scalar indicating the size of the output tensor.
    * @return Result as a new tensor.
    */
  def sparseSegmentSum[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
      data: Tensor[T],
      indices: Tensor[I1],
      segmentIndices: Tensor[Int],
      numSegments: Tensor[I2] = null
  ): Tensor[T] = {
    if (numSegments == null) {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sparseSegmentSum(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle))
    } else {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sparseSegmentSumWithNumSegments(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle,
        numSegments.nativeHandle))
    }
  }

  /** $OpDocMathSparseSegmentMean
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  numSegments    Optional scalar indicating the size of the output tensor.
    * @return Result as a new tensor.
    */
  def sparseSegmentMean[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
      data: Tensor[T],
      indices: Tensor[I1],
      segmentIndices: Tensor[Int],
      numSegments: Tensor[I2] = null
  ): Tensor[T] = {
    if (numSegments == null) {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sparseSegmentMean(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle))
    } else {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sparseSegmentMeanWithNumSegments(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle,
        numSegments.nativeHandle))
    }
  }

  /** $OpDocMathSparseSegmentSumSqrtN
    *
    * @group MathOps
    * @param  data           Data tensor.
    * @param  indices        One-dimensional tensor with rank equal to that of `segmentIndices`.
    * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
    * @param  numSegments    Optional scalar indicating the size of the output tensor.
    * @return Result as a new tensor.
    */
  def sparseSegmentSumSqrtN[T: TF : IsReal, I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
      data: Tensor[T],
      indices: Tensor[I1],
      segmentIndices: Tensor[Int],
      numSegments: Tensor[I2] = null
  ): Tensor[T] = {
    if (numSegments == null) {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sparseSegmentSqrtN(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle))
    } else {
      Tensor.fromNativeHandle[T](NativeTensorOpsMath.sparseSegmentSqrtNWithNumSegments(
        executionContext.value.nativeHandle, data.nativeHandle, indices.nativeHandle, segmentIndices.nativeHandle,
        numSegments.nativeHandle))
    }
  }

  //endregion Segment Ops

  //region Matrix Ops

  /** $OpDocMathDiag
    *
    * @group MathOps
    * @param  diagonal Diagonal values, represented as a rank-`K` tensor, where `K` can be at most `3`.
    * @return Result as a new tensor.
    */
  def diag[T: TF : IsNotQuantized](diagonal: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.diag(
      executionContext.value.nativeHandle, diagonal.nativeHandle))
  }

  /** $OpDocMathDiagPart
    *
    * @group MathOps
    * @param  input Rank-`K` input tensor, where `K` is either `2`, `4`, or `6`.
    * @return Result as a new tensor.
    */
  def diagPart[T: TF : IsNotQuantized](input: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.diagPart(
      executionContext.value.nativeHandle, input.nativeHandle))
  }

  /** $OpDocMathMatrixDiag
    *
    * @group MathOps
    * @param  diagonal Rank-`K` input tensor, where `K >= 1`.
    * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its 
    *         last dimension duplicated.
    */
  def matrixDiag[T: TF](diagonal: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.matrixDiag(
      executionContext.value.nativeHandle, diagonal.nativeHandle))
  }

  /** $OpDocMathMatrixSetDiag
    *
    * @group MathOps
    * @param  input    Rank-`K+1` tensor, where `K >= 2`.
    * @param  diagonal Rank-`K` tensor, where `K >= 1`.
    * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `input`.
    */
  def matrixSetDiag[T: TF](input: Tensor[T], diagonal: Tensor[T]): Tensor[T] = {
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
  def matrixDiagPart[T: TF](input: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.matrixDiagPart(
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
  def matrixBandPart[T: TF, I: TF : IsInt32OrInt64](
      input: Tensor[T],
      numSubDiagonals: Tensor[I],
      numSuperDiagonals: Tensor[I]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.matrixBandPart(
      executionContext.value.nativeHandle, input.nativeHandle, numSubDiagonals.nativeHandle,
      numSuperDiagonals.nativeHandle))
  }

  /** $OpDocMathTrace
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def trace[T: TF : IsNumeric](input: Tensor[T]): Tensor[T] = {
    sum(matrixDiagPart(input), axes = -1)
  }

  /** $OpDocMathScalarMul
    *
    * @group MathOps
    * @param  scalar Scalar tensor.
    * @param  tensor Tensor to multiply the scalar tensor with.
    * @return Result as a new tensor.
    */
  def scalarMul[T: TF : IsNotQuantized, TL[A] <: TensorLike[A]](
      scalar: Tensor[T],
      tensor: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[T] = {
    ev.applyUnary(tensor, t => multiply(scalar, t))
  }

  // TODO: [TENSORS] The following type constraints are wrong.

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
  def matmul[T: TF : IsNotQuantized](
      a: Tensor[T],
      b: Tensor[T],
      transposeA: Boolean = false,
      transposeB: Boolean = false,
      conjugateA: Boolean = false,
      conjugateB: Boolean = false,
      aIsSparse: Boolean = false,
      bIsSparse: Boolean = false
  ): Tensor[T] = {
    val sparseMatMulDataTypes = Set[DataType[_]](BFLOAT16, FLOAT32)
    if (!aIsSparse && !bIsSparse && (a.rank == -1 || a.rank > 2) && (b.rank == -1 || b.rank > 2)) {
      // "BatchMatMul" does not support transpose, so we conjugate the matrix and use adjoint instead.
      // The "conj" op is a no-op for real matrices.
      val (x, adjointX) = transposeConjugateToAdjoint(a, transposeA, conjugateA)
      val (y, adjointY) = transposeConjugateToAdjoint(b, transposeB, conjugateB)
      Tensor.fromNativeHandle(NativeTensorOpsMath.batchMatMul(
        executionContext.value.nativeHandle, x.nativeHandle, y.nativeHandle, adjointX, adjointY))
    } else if ((aIsSparse || bIsSparse) &&
        sparseMatMulDataTypes.contains(a.dataType) &&
        sparseMatMulDataTypes.contains(b.dataType)) {
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

  private[this] def transposeConjugateToAdjoint[T: TF : IsNotQuantized](
      tensor: Tensor[T],
      transpose: Boolean,
      conj: Boolean
  ): (Tensor[T], Boolean) = {
    // TODO: [TYPES] These runtime checks are not elegant.
    (transpose, conj) match {
      case (false, false) => (tensor, false)
      case (false, true) if tensor.dataType == COMPLEX64 =>
        (conjugate(tensor.asInstanceOf[Tensor[ComplexFloat]]).asInstanceOf[Tensor[T]], false)
      case (false, true) if tensor.dataType == COMPLEX128 =>
        (conjugate(tensor.asInstanceOf[Tensor[ComplexDouble]]).asInstanceOf[Tensor[T]], false)
      case (false, true) => (tensor, false)
      case (true, false) if tensor.dataType == COMPLEX64 =>
        (conjugate(tensor.asInstanceOf[Tensor[ComplexFloat]]).asInstanceOf[Tensor[T]], true)
      case (true, false) if tensor.dataType == COMPLEX128 =>
        (conjugate(tensor.asInstanceOf[Tensor[ComplexDouble]]).asInstanceOf[Tensor[T]], true)
      case (true, false) => (tensor, true)
      case (true, true) => (tensor, true)
    }
  }

  private[this] def transposeConjugateToTranspose[T: TF : IsNotQuantized](
      tensor: Tensor[T],
      transpose: Boolean,
      conj: Boolean
  ): (Tensor[T], Boolean) = {
    // TODO: [TYPES] These runtime checks are not elegant.
    (transpose, conj) match {
      case (false, false) => (tensor, false)
      case (false, true) if tensor.dataType == COMPLEX64 =>
        (conjugate(tensor.asInstanceOf[Tensor[ComplexFloat]]).asInstanceOf[Tensor[T]], false)
      case (false, true) if tensor.dataType == COMPLEX128 =>
        (conjugate(tensor.asInstanceOf[Tensor[ComplexDouble]]).asInstanceOf[Tensor[T]], false)
      case (false, true) => (tensor, false)
      case (true, false) => (tensor, true)
      case (true, true) if tensor.dataType == COMPLEX64 =>
        (conjugate(tensor.asInstanceOf[Tensor[ComplexFloat]]).asInstanceOf[Tensor[T]], true)
      case (true, true) if tensor.dataType == COMPLEX128 =>
        (conjugate(tensor.asInstanceOf[Tensor[ComplexDouble]]).asInstanceOf[Tensor[T]], true)
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
  def cross[T: TF : IsReal](a: Tensor[T], b: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.cross(
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
  def tensorDot[T: TF : IsNotQuantized](
      a: Tensor[T],
      b: Tensor[T],
      numAxes: Tensor[Int]
  ): Tensor[T] = {
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
  def tensorDot[T: TF : IsNotQuantized](
      a: Tensor[T],
      b: Tensor[T],
      axesA: Tensor[Int],
      axesB: Tensor[Int]
  ): Tensor[T] = {
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
    def tensorDotReshape(a: Tensor[T], axes: Tensor[Int], flipped: Boolean = false): (Tensor[T], Tensor[Int]) = {
      val shapeA = Basic.shape(a)
      val rankA = Basic.rank(a)
      val mappedAxes = ((axes >= 0).toInt * axes) + ((axes < 0).toInt * (axes + rankA.toInt))
      val (free, _) = Basic.listDiff(Math.range(0, rankA), mappedAxes, indicesDataType = Int)
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
      (reshapedA, freeAxes.toInt)
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
  def complexFloat(real: Tensor[Float], imag: Tensor[Float]): Tensor[ComplexFloat] = {
    Tensor.fromNativeHandle[ComplexFloat](NativeTensorOpsMath.complex(
      executionContext.value.nativeHandle, real.nativeHandle, imag.nativeHandle, COMPLEX64.cValue))
  }

  /** $OpDocMathComplex
    *
    * @group MathOps
    * @param  real Tensor containing the real component.
    * @param  imag Tensor containing the imaginary component.
    * @return Result as a new tensor.
    */
  def complexDouble(real: Tensor[Double], imag: Tensor[Double]): Tensor[ComplexDouble] = {
    Tensor.fromNativeHandle[ComplexDouble](NativeTensorOpsMath.complex(
      executionContext.value.nativeHandle, real.nativeHandle, imag.nativeHandle, COMPLEX128.cValue))
  }

  /** $OpDocMathReal
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def realFloat[TL[A] <: TensorLike[A]](
      input: TL[ComplexFloat],
      name: String = "Real"
  )(implicit
      ev: TensorOps.Aux[TL, ComplexFloat]
  ): TL[Float] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[Float](NativeTensorOpsMath.real(
        executionContext.value.nativeHandle, t.nativeHandle, FLOAT32.cValue))
    })
  }

  /** $OpDocMathReal
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def realDouble[TL[A] <: TensorLike[A]](
      input: TL[ComplexDouble],
      name: String = "Real"
  )(implicit
      ev: TensorOps.Aux[TL, ComplexDouble]
  ): TL[Double] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[Double](NativeTensorOpsMath.real(
        executionContext.value.nativeHandle, t.nativeHandle, FLOAT64.cValue))
    })
  }

  /** $OpDocMathImag
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def imagFloat[TL[A] <: TensorLike[A]](
      input: TL[ComplexFloat],
      name: String = "Imag"
  )(implicit
      ev: TensorOps.Aux[TL, ComplexFloat]
  ): TL[Float] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[Float](NativeTensorOpsMath.imag(
        executionContext.value.nativeHandle, t.nativeHandle, FLOAT32.cValue))
    })
  }

  /** $OpDocMathImag
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def imagDouble[TL[A] <: TensorLike[A]](
      input: TL[ComplexDouble],
      name: String = "Imag"
  )(implicit
      ev: TensorOps.Aux[TL, ComplexDouble]
  ): TL[Double] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[Double](NativeTensorOpsMath.imag(
        executionContext.value.nativeHandle, t.nativeHandle, FLOAT64.cValue))
    })
  }

  /** $OpDocMathAbs
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def magnitudeFloat[TL[A] <: TensorLike[A]](
      input: TL[ComplexFloat],
      name: String = "Magnitude"
  )(implicit
      ev: TensorOps.Aux[TL, ComplexFloat]
  ): TL[Float] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[Float](NativeTensorOpsMath.complexAbs(
        executionContext.value.nativeHandle, t.nativeHandle, FLOAT32.cValue))
    })
  }

  /** $OpDocMathAbs
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def magnitudeDouble[TL[A] <: TensorLike[A]](
      input: TL[ComplexDouble],
      name: String = "Magnitude"
  )(implicit
      ev: TensorOps.Aux[TL, ComplexDouble]
  ): TL[Double] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[Double](NativeTensorOpsMath.complexAbs(
        executionContext.value.nativeHandle, t.nativeHandle, FLOAT64.cValue))
    })
  }

  /** $OpDocMathAngle
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def angleFloat[TL[A] <: TensorLike[A]](
      input: TL[ComplexFloat],
      name: String = "Angle"
  )(implicit
      ev: TensorOps.Aux[TL, ComplexFloat]
  ): TL[Float] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[Float](NativeTensorOpsMath.angle(
        executionContext.value.nativeHandle, t.nativeHandle, FLOAT32.cValue))
    })
  }

  /** $OpDocMathAngle
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def angleDouble[TL[A] <: TensorLike[A]](
      input: TL[ComplexDouble],
      name: String = "Angle"
  )(implicit
      ev: TensorOps.Aux[TL, ComplexDouble]
  ): TL[Double] = {
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[Double](NativeTensorOpsMath.angle(
        executionContext.value.nativeHandle, t.nativeHandle, FLOAT64.cValue))
    })
  }

  /** $OpDocMathConjugate
    *
    * @group MathOps
    * @param  input Input tensor.
    * @return Result as a new tensor.
    */
  def conjugate[T: TF, TL[A] <: TensorLike[A]](
      input: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[T] = {
    ev.applyUnary(input, t => {
      if (t.dataType.isComplex) {
        Tensor.fromNativeHandle[T](NativeTensorOpsMath.conj(
          executionContext.value.nativeHandle, t.nativeHandle))
      } else {
        t
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
    * @param  boundaries Sorted sequence of numbers specifying the boundaries of the buckets.
    * @return Result as a new tensor.
    */
  def bucketize[T: TF : IsInt32OrInt64OrFloat32OrFloat64](
      input: Tensor[T],
      boundaries: Seq[Float]
  ): Tensor[T] = {
    Tensor.fromNativeHandle[T](NativeTensorOpsMath.bucketize(
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
  def zerosFraction[T: TF : IsNumeric](
      input: Tensor[T]
  ): Tensor[Float] = {
    mean(equal(input, Tensor.zeros(input.dataType, Shape())).toFloat)
  }

  //endregion Other Ops
}

object Math extends Math {
  private[tensors] trait Implicits {
    implicit def tensorConvertibleToMathOps[T: TF, TC](value: TC)(implicit
        f: TC => Tensor[T]
    ): MathOps[T] = {
      new MathOps(f(value))
    }

    implicit def tensorConvertibleToFloatMathOps[TC](
        value: TC
    )(implicit f: TC => Tensor[Float]): FloatMathOps = {
      new FloatMathOps(f(value))
    }

    implicit def tensorConvertibleToDoubleMathOps[TC](
        value: TC
    )(implicit f: TC => Tensor[Double]): DoubleMathOps = {
      new DoubleMathOps(f(value))
    }

    implicit def tensorConvertibleToComplexFloatMathOps[TC](
        value: TC
    )(implicit f: TC => Tensor[ComplexFloat]): ComplexFloatMathOps = {
      new ComplexFloatMathOps(f(value))
    }

    implicit def tensorConvertibleToComplexDoubleMathOps[TC](
        value: TC
    )(implicit f: TC => Tensor[ComplexDouble]): ComplexDoubleMathOps = {
      new ComplexDoubleMathOps(f(value))
    }

    implicit class MathOps[T](val tensor: Tensor[T]) {
      protected implicit val evTTF: TF[T] = {
        TF.fromDataType(tensor.dataType)
      }

      /** $OpDocMathSelect
        *
        * @group MathOps
        * @param  x Tensor which may have the same shape as `condition`. If `condition` has rank `1`, then `t` may have
        *           a higher rank, but its first dimension must match the size of `condition`.
        * @param  y Tensor with the same data type and shape as `t`.
        * @return Created op output.
        */
      def select[R: TF](
          x: Tensor[R],
          y: Tensor[R]
      )(implicit ev: T =:= Boolean): Tensor[R] = {
        Math.select(tensor.asInstanceOf[Tensor[Boolean]], x, y)
      }

      //region Operators

      /** $OpDocMathLogicalNot
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def unary_!(implicit ev: T =:= Boolean): Tensor[Boolean] = {
        logicalNot
      }

      /** $OpDocMathLogicalAnd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def &&(other: Tensor[Boolean])(implicit ev: T =:= Boolean): Tensor[Boolean] = {
        logicalAnd(other)
      }

      /** $OpDocMathLogicalOr
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ||(other: Tensor[Boolean])(implicit ev: T =:= Boolean): Tensor[Boolean] = {
        logicalOr(other)
      }

      /** $OpDocMathEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ===(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.equal(tensor, other)
      }

      /** $OpDocMathNotEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def =!=(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.notEqual(tensor, other)
      }

      /** $OpDocMathLess
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def <(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.less(tensor, other)
      }

      /** $OpDocMathLessEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def <=(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.lessEqual(tensor, other)
      }

      /** $OpDocMathGreater
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def >(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.greater(tensor, other)
      }

      /** $OpDocMathGreaterEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def >=(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.greaterEqual(tensor, other)
      }

      /** $OpDocMathNegate
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def unary_-(implicit
          ev: IsNotQuantized[T]
      ): Tensor[T] = {
        Math.negate(tensor.castTo[T])
      }

      /** $OpDocMathAdd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def +(other: Tensor[T])(implicit
          ev: IsNotQuantized[T]
      ): Tensor[T] = {
        Math.add(tensor, other)
      }

      /** $OpDocMathSubtract
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def -(other: Tensor[T])(implicit
          ev: IsNotQuantized[T]
      ): Tensor[T] = {
        Math.subtract(tensor, other)
      }

      /** $OpDocMathMultiply
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def *(other: Tensor[T])(implicit
          ev: IsNotQuantized[T]
      ): Tensor[T] = {
        Math.multiply(tensor, other)
      }

      /** $OpDocMathDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def /(other: Tensor[T])(implicit
          ev: IsNotQuantized[T]
      ): Tensor[T] = {
        if (tensor.dataType.isFloatingPoint || tensor.dataType.isComplex)
          Math.divide(tensor, other)
        else
          Math.truncateDivide(tensor, other)
      }

      /** $OpDocMathMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def %(other: Tensor[T])(implicit
          ev: IsNotQuantized[T]
      ): Tensor[T] = {
        Math.mod(tensor, other)
      }

      /** $OpDocMathPow
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def **(other: Tensor[T])(implicit
          ev: IsNotQuantized[T]
      ): Tensor[T] = {
        Math.pow(tensor, other)
      }

      /** $OpDocMathPow
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ^(other: Tensor[T])(implicit
          ev: IsNotQuantized[T]
      ): Tensor[T] = {
        Math.pow(tensor, other)
      }

      //endregion Operators

      //region Unary Ops

      /** $OpDocMathAbs
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def abs(implicit ev: IsReal[T]): Tensor[T] = {
        Math.abs(tensor)
      }

      /** $OpDocMathNegate
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def negate(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.negate(tensor)
      }

      /** $OpDocMathReciprocal
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def reciprocal(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.reciprocal(tensor)
      }

      /** $OpDocMathSquare
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def square(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.square(tensor)
      }

      /** $OpDocMathSqrt
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sqrt(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.sqrt(tensor)
      }

      /** $OpDocMathRsqrt
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def rsqrt(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.rsqrt(tensor)
      }

      /** $OpDocMathExp
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def exp(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.exp(tensor)
      }

      /** $OpDocMathExpm1
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def expm1(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.expm1(tensor)
      }

      /** $OpDocMathLog
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def log(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.log(tensor)
      }

      /** $OpDocMathLog1p
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def log1p(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.log1p(tensor)
      }

      /** $OpDocMathSin
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sin(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.sin(tensor)
      }

      /** $OpDocMathCos
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def cos(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.cos(tensor)
      }

      /** $OpDocMathTan
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def tan(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.tan(tensor)
      }

      /** $OpDocMathAsin
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def asin(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.asin(tensor)
      }

      /** $OpDocMathAcos
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def acos(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.acos(tensor)
      }

      /** $OpDocMathAtan
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def atan(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.atan(tensor)
      }

      /** $OpDocMathSinh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sinh(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.sinh(tensor)
      }

      /** $OpDocMathCosh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def cosh(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.cosh(tensor)
      }

      /** $OpDocMathTanh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def tanh(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.tanh(tensor)
      }

      /** $OpDocMathAsinh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def asinh(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.asinh(tensor)
      }

      /** $OpDocMathAcosh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def acosh(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.acosh(tensor)
      }

      /** $OpDocMathAtanh
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def atanh(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.atanh(tensor)
      }

      /** $OpDocMathLogGamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logGamma(implicit ev: IsFloat32OrFloat64[T]): Tensor[T] = {
        Math.logGamma(tensor)
      }

      /** $OpDocMathDigamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def digamma(implicit ev: IsFloat32OrFloat64[T]): Tensor[T] = {
        Math.digamma(tensor)
      }

      /** $OpDocMathErf
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def erf(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.erf(tensor)
      }

      /** $OpDocMathErfc
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def erfc(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.erfc(tensor)
      }

      /** $OpDocMathSigmoid
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sigmoid(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.sigmoid(tensor)
      }

      /** $OpDocMathLogSigmoid
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logSigmoid(implicit ev: IsDecimal[T]): Tensor[T] = {
        Math.logSigmoid(tensor)
      }

      /** $OpDocMathSign
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def sign(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.sign(tensor)
      }

      /** $OpDocMathRound
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def round(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.round(tensor)
      }

      /** $OpDocMathRoundInt
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def roundInt(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Tensor[T] = {
        Math.roundInt(tensor)
      }

      /** $OpDocMathFloor
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def floor(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Tensor[T] = {
        Math.floor(tensor)
      }

      /** $OpDocMathCeil
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def ceil(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Tensor[T] = {
        Math.ceil(tensor)
      }

      /** $OpDocMathIsNaN
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def isNaN(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Tensor[Boolean] = {
        Math.isNaN(tensor)
      }

      /** $OpDocMathIsInf
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def isInf(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Tensor[Boolean] = {
        Math.isInf(tensor)
      }

      /** $OpDocMathIsFinite
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def isFinite(implicit ev: IsFloat16OrFloat32OrFloat64[T]): Tensor[Boolean] = {
        Math.isFinite(tensor)
      }

      //endregion Unary Ops

      //region Binary Ops

      /** $OpDocMathAdd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def add(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.add(tensor, other)
      }

      /** $OpDocMathSubtract
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def subtract(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.subtract(tensor, other)
      }

      /** $OpDocMathMultiply
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def multiply(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.multiply(tensor, other)
      }

      /** $OpDocMathDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def divide(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.divide(tensor, other)
      }

      /** $OpDocMathFloorDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      @deprecated("Use `truncateDivide` instead.", "0.1")
      def floorDivide(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.floorDivide(tensor, other)
      }

      /** $OpDocMathTruncateDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def truncateDivide(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.truncateDivide(tensor, other)
      }

      /** $OpDocMathRealDivide
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def realDivide(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.realDivide(tensor, other)
      }

      /** $OpDocMathSquaredDifference
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def squaredDifference(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.squaredDifference(tensor, other)
      }

      /** $OpDocMathMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def mod(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.mod(tensor, other)
      }

      /** $OpDocMathFloorMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def floorMod(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.floorMod(tensor, other)
      }

      /** $OpDocMathTruncateMod
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def truncateMod(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.truncateMod(tensor, other)
      }

      /** $OpDocMathPow
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def pow(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.pow(tensor, other)
      }

      /** $OpDocMathIgammac
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def igammac(other: Tensor[T])(implicit ev: IsFloat32OrFloat64[T]): Tensor[T] = {
        Math.igammac(tensor, other)
      }

      /** $OpDocMathIgamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def igamma(other: Tensor[T])(implicit ev: IsFloat32OrFloat64[T]): Tensor[T] = {
        Math.igamma(tensor, other)
      }

      /** $OpDocMathZeta
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def zeta(other: Tensor[T])(implicit ev: IsFloat32OrFloat64[T]): Tensor[T] = {
        Math.zeta(tensor, other)
      }

      /** $OpDocMathPolygamma
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def polygamma(other: Tensor[T])(implicit ev: IsFloat32OrFloat64[T]): Tensor[T] = {
        Math.polygamma(tensor, other)
      }

      /** $OpDocMathAtan2
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def atan2(other: Tensor[T])(implicit ev: IsFloat32OrFloat64[T]): Tensor[T] = {
        Math.atan2(tensor, other)
      }

      /** $OpDocMathMinimum
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def minimum(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.minimum(tensor, other)
      }

      /** $OpDocMathMaximum
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def maximum(other: Tensor[T])(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.maximum(tensor, other)
      }

      //endregion Binary Ops

      //region Logical Ops

      /** $OpDocMathLogicalNot
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalNot(implicit ev: T =:= Boolean): Tensor[Boolean] = {
        Math.logicalNot(tensor.asInstanceOf[Tensor[Boolean]])
      }

      /** $OpDocMathLogicalAnd
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalAnd(other: Tensor[Boolean])(implicit ev: T =:= Boolean): Tensor[Boolean] = {
        Math.logicalAnd(tensor.asInstanceOf[Tensor[Boolean]], other)
      }

      /** $OpDocMathLogicalOr
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalOr(other: Tensor[Boolean])(implicit ev: T =:= Boolean): Tensor[Boolean] = {
        Math.logicalOr(tensor.asInstanceOf[Tensor[Boolean]], other)
      }

      /** $OpDocMathLogicalXOr
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def logicalXOr(other: Tensor[Boolean])(implicit ev: T =:= Boolean): Tensor[Boolean] = {
        Math.logicalXOr(tensor.asInstanceOf[Tensor[Boolean]], other)
      }

      //endregion Logical Ops

      //region Comparison Ops

      /** $OpDocMathEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def equal(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.equal(tensor, other)
      }

      /** $OpDocMathNotEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def notEqual(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.notEqual(tensor, other)
      }

      /** $OpDocMathApproximatelyEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def approximatelyEqual(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.approximatelyEqual(tensor, other)
      }

      /** $OpDocMathLess
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def less(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.less(tensor, other)
      }

      /** $OpDocMathLessEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def lessEqual(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.lessEqual(tensor, other)
      }

      /** $OpDocMathGreater
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def greater(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.greater(tensor, other)
      }

      /** $OpDocMathGreaterEqual
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def greaterEqual(other: Tensor[T])(implicit ev: IsNumeric[T]): Tensor[Boolean] = {
        Math.greaterEqual(tensor, other)
      }

      //endregion Comparison Ops

      //region Reduction Ops

      /** $OpDocMathSum
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def sum[I: IntDefault : TF : IsInt32OrInt64](
          axes: Tensor[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.sum(tensor, axes, keepDims)
      }

      /** $OpDocMathMean
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def mean[I: IntDefault : TF : IsInt32OrInt64](
          axes: Tensor[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.mean(tensor, axes, keepDims)
      }

      /** $OpDocMathProd
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def prod[I: IntDefault : TF : IsInt32OrInt64](
          axes: Tensor[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.prod(tensor, axes, keepDims)
      }

      /** $OpDocMathMin
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def min[I: IntDefault : TF : IsInt32OrInt64](
          axes: Tensor[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.min(tensor, axes, keepDims)
      }

      /** $OpDocMathMax
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def max[I: IntDefault : TF : IsInt32OrInt64](
          axes: Tensor[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.max(tensor, axes, keepDims)
      }

      /** $OpDocMathAll
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def all[I: IntDefault : TF : IsInt32OrInt64](
          axes: Tensor[I] = null,
          keepDims: Boolean = false
      )(implicit ev: T =:= Boolean): Tensor[Boolean] = {
        Math.all(tensor.asInstanceOf[Tensor[Boolean]], axes, keepDims)
      }

      /** $OpDocMathAny
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def any[I: IntDefault : TF : IsInt32OrInt64](
          axes: Tensor[I] = null,
          keepDims: Boolean = false
      )(implicit ev: T =:= Boolean): Tensor[Boolean] = {
        Math.any(tensor.asInstanceOf[Tensor[Boolean]], axes, keepDims)
      }

      /** $OpDocMathLogSumExp
        *
        * @group MathOps
        * @param  axes     Integer sequence containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def logSumExp(
          axes: Seq[Int] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.logSumExp(tensor, axes, keepDims)
      }

      /** $OpDocMathCountNonZero
        *
        * @group MathOps
        * @param  axes     Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  keepDims If `true`, retain the reduced axes.
        * @return Result as a new tensor.
        */
      def countNonZero[I: IntDefault : TF : IsInt32OrInt64](
          axes: Tensor[I] = null,
          keepDims: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[Long] = {
        Math.countNonZero(tensor, axes, keepDims)
      }

      /** $OpDocMathArgmin
        *
        * @group MathOps
        * @param  axes Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @return Result as a new tensor.
        */
      def argmin[I: TF : IsInt32OrInt64](
          axes: Tensor[I]
      )(implicit ev: IsNotQuantized[T]): Tensor[Long] = {
        Math.argmin(tensor, axes)
      }

      /** $OpDocMathArgmin
        *
        * @group MathOps
        * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  outputDataType Data type for the output tensor.
        * @return Result as a new tensor.
        */
      def argmin[I: TF : IsInt32OrInt64, IR: TF : IsInt32OrInt64](
          axes: Tensor[I],
          outputDataType: DataType[IR]
      )(implicit ev: IsNotQuantized[T]): Tensor[IR] = {
        Math.argmin(tensor, axes, outputDataType)
      }

      /** $OpDocMathArgmax
        *
        * @group MathOps
        * @param  axes Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @return Result as a new tensor.
        */
      def argmax[I: TF : IsInt32OrInt64](
          axes: Tensor[I]
      )(implicit ev: IsNotQuantized[T]): Tensor[Long] = {
        Math.argmax(tensor, axes)
      }

      /** $OpDocMathArgmax
        *
        * @group MathOps
        * @param  axes           Integer tensor containing the axes to reduce. If `null`, then all axes are reduced.
        * @param  outputDataType Data type for the output tensor.
        * @return Result as a new tensor.
        */
      def argmax[I: TF : IsInt32OrInt64, IR: TF : IsInt32OrInt64](
          axes: Tensor[I],
          outputDataType: DataType[IR]
      )(implicit ev: IsNotQuantized[T]): Tensor[IR] = {
        Math.argmax(tensor, axes, outputDataType)
      }

      /** $OpDocMathCumsum
        *
        * @group MathOps
        * @param  axis      Tensor containing the axis along which to perform the cumulative sum.
        * @param  exclusive Boolean value indicating whether to perform an exclusive cumulative sum.
        * @param  reverse   Boolean value indicating whether to perform a reverse cumulative sum.
        * @return Result as a new tensor.
        */
      def cumsum[I: TF : IsInt32OrInt64](
          axis: Tensor[I],
          exclusive: Boolean = false,
          reverse: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
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
      def cumprod[I: TF : IsInt32OrInt64](
          axis: Tensor[I],
          exclusive: Boolean = false,
          reverse: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.cumprod(tensor, axis, exclusive, reverse)
      }

      //endregion Reduction Ops

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
      def binCount[R: TF : IsInt32OrInt64OrFloat32OrFloat64](
          dataType: DataType[R],
          weights: Tensor[R] = null,
          minLength: Tensor[Int] = null,
          maxLength: Tensor[Int] = null
      )(implicit ev: T =:= Int): Tensor[R] = {
        Math.binCount(tensor.asInstanceOf[Tensor[Int]], dataType, weights, minLength, maxLength)
      }

      //region Segment Ops

      /** $OpDocMathSegmentSum
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentSum[I: TF : IsInt32OrInt64](
          segmentIndices: Tensor[I]
      )(implicit ev: IsNumeric[T]): Tensor[T] = {
        Math.segmentSum(tensor, segmentIndices)
      }

      /** $OpDocMathSegmentMean
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentMean[I: TF : IsInt32OrInt64](
          segmentIndices: Tensor[I]
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.segmentMean(tensor, segmentIndices)
      }

      /** $OpDocMathSegmentProd
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentProd[I: TF : IsInt32OrInt64](
          segmentIndices: Tensor[I]
      )(implicit ev: IsNumeric[T]): Tensor[T] = {
        Math.segmentProd(tensor, segmentIndices)
      }

      /** $OpDocMathSegmentMin
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentMin[I: TF : IsInt32OrInt64](
          segmentIndices: Tensor[I]
      )(implicit ev: IsReal[T]): Tensor[T] = {
        Math.segmentMin(tensor, segmentIndices)
      }

      /** $OpDocMathSegmentMax
        *
        * @group MathOps
        * @param  segmentIndices Segment indices. Values should be sorted and can be repeated.
        * @return Result as a new tensor.
        */
      def segmentMax[I: TF : IsInt32OrInt64](
          segmentIndices: Tensor[I]
      )(implicit ev: IsReal[T]): Tensor[T] = {
        Math.segmentMax(tensor, segmentIndices)
      }

      /** $OpDocMathUnsortedSegmentSum
        *
        * @group MathOps
        * @param  segmentIndices Segment indices.
        * @param  segmentsNumber Number of segments.
        * @return Result as a new tensor.
        */
      def unsortedSegmentSum[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
          segmentIndices: Tensor[I1],
          segmentsNumber: Tensor[I2]
      )(implicit ev: IsNumeric[T]): Tensor[T] = {
        Math.unsortedSegmentSum(tensor, segmentIndices, segmentsNumber)
      }

      // TODO: [TENSORS] Missing 'unsortedSegmentMean'.
      // TODO: [TENSORS] Missing 'unsortedSegmentProd'.
      // TODO: [TENSORS] Missing 'unsortedSegmentMin'.

      /** $OpDocMathUnsortedSegmentMax
        *
        * @group MathOps
        * @param  segmentIndices Segment indices.
        * @param  segmentsNumber Number of segments.
        * @return Result as a new tensor.
        */
      def unsortedSegmentMax[I1: TF : IsInt32OrInt64, I2: TF : IsInt32OrInt64](
          segmentIndices: Tensor[I1],
          segmentsNumber: Tensor[I2]
      )(implicit ev: IsReal[T]): Tensor[T] = {
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
      def sparseSegmentSum[I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
          indices: Tensor[I1],
          segmentIndices: Tensor[Int],
          numSegments: Tensor[I2] = null
      )(implicit ev: IsReal[T]): Tensor[T] = {
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
      def sparseSegmentMean[I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
          indices: Tensor[I1],
          segmentIndices: Tensor[Int],
          numSegments: Tensor[I2] = null
      )(implicit ev: IsReal[T]): Tensor[T] = {
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
      def sparseSegmentSumSqrtN[I1: TF : IsInt32OrInt64, I2: IntDefault : TF : IsInt32OrInt64](
          indices: Tensor[I1],
          segmentIndices: Tensor[Int],
          numSegments: Tensor[I2] = null
      )(implicit ev: IsReal[T]): Tensor[T] = {
        Math.sparseSegmentSumSqrtN(tensor, indices, segmentIndices, numSegments)
      }

      //endregion Segment Ops

      //region Matrix Ops

      /** $OpDocMathDiag
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def diag(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.diag(tensor)
      }

      /** $OpDocMathDiagPart
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def diagPart(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.diagPart(tensor)
      }

      /** $OpDocMathMatrixDiag
        *
        * @group MathOps
        * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `diagonal`, with its
        *         last dimension duplicated.
        */
      def matrixDiag: Tensor[T] = {
        Math.matrixDiag(tensor)
      }

      /** $OpDocMathMatrixSetDiag
        *
        * @group MathOps
        * @param  diagonal Rank-`K` tensor, where `K >= 1`.
        * @return Result as a new tensor with rank equal to `K + 1` and shape equal to the shape of `input`.
        */
      def matrixSetDiag(diagonal: Tensor[T]): Tensor[T] = {
        Math.matrixSetDiag(tensor, diagonal)
      }

      /** $OpDocMathMatrixDiagPart
        *
        * @group MathOps
        * @return Result as a new tensor containing the diagonal(s) and having shape equal to
        *         `input.shape[:-2] + [min(input.shape[-2:])]`.
        */
      def matrixDiagPart: Tensor[T] = {
        Math.matrixDiagPart(tensor)
      }

      /** $OpDocMathMatrixBandPart
        *
        * @group MathOps
        * @param  numSubDiagonals   Scalar tensor that contains the number of sub-diagonals to keep. If negative,
        *                           the entire lower triangle is kept.
        * @param  numSuperDiagonals Scalar tensor that contains the number of super-diagonals to keep. If negative,
        *                           the entire upper triangle is kept.
        * @return Result as a new tensor containing the expected banded tensor and has rank `K` and same shape as `input`.
        */
      def matrixBandPart[I: TF : IsInt32OrInt64](
          numSubDiagonals: Tensor[I],
          numSuperDiagonals: Tensor[I]
      ): Tensor[T] = {
        Math.matrixBandPart(tensor, numSubDiagonals, numSuperDiagonals)
      }

      /** $OpDocMathTrace
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def trace(implicit ev: IsNumeric[T]): Tensor[T] = {
        Math.trace(tensor)
      }

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
          other: Tensor[T],
          transposeA: Boolean = false,
          transposeB: Boolean = false,
          conjugateA: Boolean = false,
          conjugateB: Boolean = false,
          aIsSparse: Boolean = false,
          bIsSparse: Boolean = false
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.matmul(tensor, other, transposeA, transposeB, conjugateA, conjugateB, aIsSparse, bIsSparse)
      }

      /** $OpDocMathCross
        *
        * @group MathOps
        * @param  other Tensor to multiply with.
        * @return Result as a new tensor.
        */
      def cross(
          other: Tensor[T]
      )(implicit ev: IsReal[T]): Tensor[T] = {
        Math.cross(tensor, other)
      }

      /** Dynamic version (i.e., where `numAxes` may be a tensor) of the `tensorDot` op.
        *
        * $OpDocMathTensorDot
        *
        * @group MathOps
        * @param  other   Tensor to contract with.
        * @param  numAxes Number of axes to contract.
        * @return Created op output.
        */
      def tensorDot(
          other: Tensor[T],
          numAxes: Tensor[Int]
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
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
      def tensorDot(
          other: Tensor[T],
          axesA: Tensor[Int],
          axesB: Tensor[Int]
      )(implicit ev: IsNotQuantized[T]): Tensor[T] = {
        Math.tensorDot(tensor, other, axesA, axesB)
      }

      //endregion Matrix Ops

      //region Complex Ops

      /** $OpDocMathConjugate
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def conjugate(implicit ev: IsComplex[T]): Tensor[T] = {
        Math.conjugate(tensor)
      }

      //endregion Complex Ops

      //region Bucketization Ops

      /** $OpDocMathBucketize
        *
        * @group MathOps
        * @param  boundaries Sorted sequence of numbers specifying the boundaries of the buckets.
        * @return Result as a new tensor.
        */
      def bucketize(
          boundaries: Seq[Float]
      )(implicit ev: IsInt32OrInt64OrFloat32OrFloat64[T]): Tensor[T] = {
        Math.bucketize(tensor, boundaries)
      }

      //endregion Bucketization Ops

      //region Other Ops

      /** $OpDocMathZerosFraction
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def zerosFraction(implicit ev: IsNumeric[T]): Tensor[Float] = {
        Math.zerosFraction(tensor)
      }

      //endregion Other Ops
    }

    implicit class FloatMathOps(val tensor: Tensor[Float]) {
      /** Creates a new complex number with the provided imaginary part.
        *
        * @param  imag Imaginary part.
        * @return Resulting complex number.
        */
      def toComplex(imag: Tensor[Float] = 0.0f): Tensor[ComplexFloat] = {
        Math.complexFloat(tensor, imag)
      }
    }

    implicit class DoubleMathOps(val tensor: Tensor[Double]) {
      /** Creates a new complex number with the provided imaginary part.
        *
        * @param  imag Imaginary part.
        * @return Resulting complex number.
        */
      def toComplex(imag: Tensor[Double] = 0.0): Tensor[ComplexDouble] = {
        Math.complexDouble(tensor, imag)
      }
    }

    implicit class ComplexFloatMathOps(val tensor: Tensor[ComplexFloat]) {
      /** $OpDocMathReal
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def real: Tensor[Float] = {
        Math.realFloat(tensor.asInstanceOf[Tensor[ComplexFloat]])
      }

      /** $OpDocMathImag
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def imag: Tensor[Float] = {
        Math.imagFloat(tensor.asInstanceOf[Tensor[ComplexFloat]])
      }

      /** $OpDocMathAbs
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def magnitude: Tensor[Float] = {
        Math.magnitudeFloat(tensor.asInstanceOf[Tensor[ComplexFloat]])
      }

      /** $OpDocMathAngle
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def angle: Tensor[Float] = {
        Math.angleFloat(tensor.asInstanceOf[Tensor[ComplexFloat]])
      }
    }

    implicit class ComplexDoubleMathOps(val tensor: Tensor[ComplexDouble]) {
      /** $OpDocMathReal
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def real: Tensor[Double] = {
        Math.realDouble(tensor.asInstanceOf[Tensor[ComplexDouble]])
      }

      /** $OpDocMathImag
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def imag: Tensor[Double] = {
        Math.imagDouble(tensor.asInstanceOf[Tensor[ComplexDouble]])
      }

      /** $OpDocMathAbs
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def magnitude: Tensor[Double] = {
        Math.magnitudeDouble(tensor.asInstanceOf[Tensor[ComplexDouble]])
      }

      /** $OpDocMathAngle
        *
        * @group MathOps
        * @return Result as a new tensor.
        */
      def angle: Tensor[Double] = {
        Math.angleDouble(tensor.asInstanceOf[Tensor[ComplexDouble]])
      }
    }
  }
}
