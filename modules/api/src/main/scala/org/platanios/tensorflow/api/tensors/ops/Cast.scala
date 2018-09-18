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

import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.generated.tensors.{Math => NativeTensorOpsMath}

/** Contains functions for executing cast-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Cast {
  /** $OpDocMathCast
    *
    * @group MathOps
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @return Result as a new tensor.
    */
  def cast[T, R, TL[TT] <: TensorLike[TT]](
      x: TL[T],
      dataType: DataType[R],
      truncate: Boolean = false
  )(implicit ev: TensorOps.Aux[TL, T]): TL[R] = {
    if (x.dataType == dataType) {
      x.asInstanceOf[TL[R]]
    } else {
      ev.applyUnary(x, t => {
        Tensor.fromNativeHandle(NativeTensorOpsMath.cast(
          executionContext.value.nativeHandle, t.nativeHandle, dataType.cValue, truncate))
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
  def bitcast[T: IsNumeric, R](input: Tensor[T], dataType: DataType[R]): Tensor[R] = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.bitcast(
      executionContext.value.nativeHandle, input.nativeHandle, dataType.cValue))
  }
}

object Cast extends Cast {
  private[tensors] trait Implicits {
    implicit class CastOps[T](val tensor: Tensor[T]) {
      /** $OpDocCastCast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def cast[R](dataType: DataType[R]): Tensor[R] = Cast.cast(tensor, dataType)

      def toStringTensor: Tensor[STRING] = cast(STRING)
      def toBoolean: Tensor[BOOLEAN] = cast(BOOLEAN)
      def toFloat16: Tensor[FLOAT16] = cast(FLOAT16)
      def toFloat32: Tensor[FLOAT32] = cast(FLOAT32)
      def toFloat64: Tensor[FLOAT64] = cast(FLOAT64)
      def toBFloat16: Tensor[BFLOAT16] = cast(BFLOAT16)
      def toComplex64: Tensor[COMPLEX64] = cast(COMPLEX64)
      def toComplex128: Tensor[COMPLEX128] = cast(COMPLEX128)
      def toInt8: Tensor[INT8] = cast(INT8)
      def toInt16: Tensor[INT16] = cast(INT16)
      def toInt32: Tensor[INT32] = cast(INT32)
      def toInt64: Tensor[INT64] = cast(INT64)
      def toUInt8: Tensor[UINT8] = cast(UINT8)
      def toUInt16: Tensor[UINT16] = cast(UINT16)
      def toUInt32: Tensor[UINT32] = cast(UINT32)
      def toUInt64: Tensor[UINT64] = cast(UINT64)
      def toQInt8: Tensor[QINT8] = cast(QINT8)
      def toQInt16: Tensor[QINT16] = cast(QINT16)
      def toQInt32: Tensor[QINT32] = cast(QINT32)
      def toQUInt8: Tensor[QUINT8] = cast(QUINT8)
      def toQUInt16: Tensor[QUINT16] = cast(QUINT16)
    }

    implicit class NumericCastOps[T: IsNumeric](val tensor: Tensor[T]) {
      /** $OpDocCastBitcast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def bitcast[R](dataType: DataType[R]): Tensor[R] = Cast.bitcast(tensor, dataType)
    }

    implicit def tensorConvertibleToCastOps[T, TC](
        value: TC
    )(implicit f: TC => Tensor[T]): CastOps[T] = {
      new CastOps(f(value))
    }

    implicit def tensorConvertibleToReducibleCastOps[T: IsNumeric, TC](
        value: TC
    )(implicit f: TC => Tensor[T]): NumericCastOps[T] = {
      new NumericCastOps(f(value))
    }
  }
}
