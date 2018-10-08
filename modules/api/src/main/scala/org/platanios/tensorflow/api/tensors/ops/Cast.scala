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

import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.jni.generated.tensors.{Math => NativeTensorOpsMath}

/** Contains functions for executing cast-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Cast {
  /** $OpDocCastCast
    *
    * @group CastOps
    *
    * @param  input Tensor to cast.
    * @tparam R Target data type.
    * @return Result as a new tensor.
    */
  private[Cast] def cast[T: TF, R: TF, TL[TT] <: TensorLike[TT]](
      input: TL[T],
      truncate: Boolean = false
  )(implicit ev: TensorOps.Aux[TL, T]): TL[R] = {
    val dataType = implicitly[TF[R]].dataType
    if (input.dataType == dataType) {
      input.asInstanceOf[TL[R]]
    } else {
      ev.applyUnary(input, t => {
        Tensor.fromNativeHandle[R](NativeTensorOpsMath.cast(
          executionContext.value.nativeHandle, t.nativeHandle, dataType.cValue, truncate))
      })
    }
  }

  // TODO: [OPS] saturateCast

  /** $OpDocCastBitcast
    *
    * @group CastOps
    *
    * @param  input Input tensor.
    * @tparam R Target data type.
    * @return Result as a new tensor.
    */
  private[Cast] def bitcast[T: IsNumeric, R: TF, TL[TT] <: TensorLike[TT]](
      input: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[R] = {
    val dataType = implicitly[TF[R]].dataType
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[R](NativeTensorOpsMath.bitcast(
        executionContext.value.nativeHandle, t.nativeHandle, dataType.cValue))
    })
  }
}

object Cast extends Cast {
  private[tensors] trait Implicits {
    implicit def tensorConvertibleToCastOps[T: TF, TC, TL[TT] <: TensorLike[TT]](
        value: TC
    )(implicit
        f: TC => TL[T],
        ev: TensorOps.Aux[TL, T]
    ): CastOps[T, TL] = {
      new CastOps(f(value))
    }

    implicit class CastOps[T: TF, TL[TT] <: TensorLike[TT]](
        val tensor: TL[T]
    )(implicit evOps: TensorOps.Aux[TL, T]) {
      /** $OpDocCastCast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R: TF]: TL[R] = {
        Cast.cast[T, R, TL](tensor)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R](dataType: DataType[R]): TL[R] = {
        implicit val evRTF: TF[R] = TF.fromDataType(dataType)
        Cast.cast[T, R, TL](tensor)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R: TF](truncate: Boolean): TL[R] = {
        Cast.cast[T, R, TL](tensor, truncate)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R](dataType: DataType[R], truncate: Boolean): TL[R] = {
        implicit val evRTF: TF[R] = TF.fromDataType(dataType)
        Cast.cast[T, R, TL](tensor, truncate)
      }

      /** $OpDocCastBitcast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def bitcastTo[R: TF](implicit ev: IsNumeric[T]): TL[R] = {
        Cast.bitcast[T, R, TL](tensor)
      }

      /** $OpDocCastBitcast
        *
        * @group CastOps
        *
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def bitcastTo[R](dataType: DataType[R])(implicit ev: IsNumeric[T]): TL[R] = {
        implicit val evRTF: TF[R] = TF.fromDataType(dataType)
        Cast.bitcast[T, R, TL](tensor)
      }

      def toStringTensor: TL[String] = tensor.castTo[String]
      def toBoolean: TL[Boolean] = tensor.castTo[Boolean]
      def toHalf: TL[Half] = tensor.castTo[Half]
      def toFloat: TL[Float] = tensor.castTo[Float]
      def toDouble: TL[Double] = tensor.castTo[Double]
      def toTruncatedHalf: TL[TruncatedHalf] = tensor.castTo[TruncatedHalf]
      def toComplexFloat: TL[ComplexFloat] = tensor.castTo[ComplexFloat]
      def toComplexDouble: TL[ComplexDouble] = tensor.castTo[ComplexDouble]
      def toByte: TL[Byte] = tensor.castTo[Byte]
      def toShort: TL[Short] = tensor.castTo[Short]
      def toInt: TL[Int] = tensor.castTo[Int]
      def toLong: TL[Long] = tensor.castTo[Long]
      def toUByte: TL[UByte] = tensor.castTo[UByte]
      def toUShort: TL[UShort] = tensor.castTo[UShort]
      def toUInt: TL[UInt] = tensor.castTo[UInt]
      def toULong: TL[ULong] = tensor.castTo[ULong]
      def toQByte: TL[QByte] = tensor.castTo[QByte]
      def toQShort: TL[QShort] = tensor.castTo[QShort]
      def toQInt: TL[QInt] = tensor.castTo[QInt]
      def toQUByte: TL[QUByte] = tensor.castTo[QUByte]
      def toQUShort: TL[QUShort] = tensor.castTo[QUShort]
      def toResource: TL[Resource] = tensor.castTo[Resource]
      def toVariant: TL[Variant] = tensor.castTo[Variant]
    }
  }
}
