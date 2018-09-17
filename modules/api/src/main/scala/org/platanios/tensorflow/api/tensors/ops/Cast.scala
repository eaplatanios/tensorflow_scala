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

import org.platanios.tensorflow.api.tensors.{Tensor, TensorLike, TensorOps, executionContext}
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
  def cast[D <: DataType, DR <: DataType, TL[DD <: DataType] <: TensorLike[DD]](x: TL[D], dataType: DR, truncate: Boolean = false)(implicit
      ev: TensorOps.Aux[TL, D]
  ): TL[DR] = {
    if (x.dataType == dataType) {
      x.asInstanceOf[TL[DR]]
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
  def bitcast[D <: ReducibleDataType, DR <: DataType](input: Tensor[D], dataType: DR): Tensor[DR] = {
    Tensor.fromNativeHandle[DR](NativeTensorOpsMath.bitcast(
      executionContext.value.nativeHandle, input.nativeHandle, dataType.cValue))
  }
}

object Cast extends Cast {
  private[tensors] trait Implicits {
    implicit class CastOps[D <: DataType](val tensor: Tensor[D]) {
      /** $OpDocCastCast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def cast[DR <: DataType](dataType: DR): Tensor[DR] = Cast.cast(tensor, dataType)

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

    implicit class ReducibleCastOps[D <: ReducibleDataType](val tensor: Tensor[D]) {
      /** $OpDocCastBitcast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def bitcast[DR <: DataType](dataType: DR): Tensor[DR] = Cast.bitcast(tensor, dataType)
    }

    implicit def tensorConvertibleToCastOps[D <: DataType, T](value: T)(implicit f: T => Tensor[D]): CastOps[D] = new CastOps(f(value))
    implicit def tensorConvertibleToReducibleCastOps[D <: ReducibleDataType, T](value: T)(implicit f: T => Tensor[D]): ReducibleCastOps[D] = new ReducibleCastOps(f(value))
  }
}
