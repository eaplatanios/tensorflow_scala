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
  private[Cast] def bitcast[T: IsNumeric, R: TF](
      input: Tensor[T]
  ): Tensor[R] = {
    val dataType = implicitly[TF[R]].dataType
    Tensor.fromNativeHandle[R](NativeTensorOpsMath.bitcast(
      executionContext.value.nativeHandle, input.nativeHandle, dataType.cValue))
  }
}

object Cast extends Cast {
  private[tensors] trait Implicits {
    implicit def tensorConvertibleToCastOps[T, TC](
        value: TC
    )(implicit f: TC => Tensor[T]): CastOps[T] = {
      new CastOps(f(value))
    }

    implicit class CastOps[T](val tensor: Tensor[T]) {
      private implicit val evTF: TF[T] = {
        TF.fromDataType(tensor.dataType)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R: TF]: Tensor[R] = {
        Cast.cast[T, R, Tensor](tensor)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R](dataType: DataType[R]): Tensor[R] = {
        implicit val evRTF: TF[R] = TF.fromDataType(dataType)
        Cast.cast[T, R, Tensor](tensor)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R: TF](truncate: Boolean): Tensor[R] = {
        Cast.cast[T, R, Tensor](tensor, truncate)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @param  dataType Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R](dataType: DataType[R], truncate: Boolean): Tensor[R] = {
        implicit val evRTF: TF[R] = TF.fromDataType(dataType)
        Cast.cast[T, R, Tensor](tensor, truncate)
      }

      /** $OpDocCastBitcast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def bitcastTo[R: TF](implicit ev: IsNumeric[T]): Tensor[R] = {
        Cast.bitcast[T, R](tensor)
      }

      /** $OpDocCastBitcast
        *
        * @group CastOps
        *
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def bitcastTo[R](dataType: DataType[R])(implicit ev: IsNumeric[T]): Tensor[R] = {
        implicit val evRTF: TF[R] = TF.fromDataType(dataType)
        Cast.bitcast[T, R](tensor)
      }

      def toStringTensor: Tensor[String] = tensor.castTo[String]
      def toBoolean: Tensor[Boolean] = tensor.castTo[Boolean]
      def toHalf: Tensor[Half] = tensor.castTo[Half]
      def toFloat: Tensor[Float] = tensor.castTo[Float]
      def toDouble: Tensor[Double] = tensor.castTo[Double]
      def toTruncatedHalf: Tensor[TruncatedHalf] = tensor.castTo[TruncatedHalf]
      def toComplexFloat: Tensor[ComplexFloat] = tensor.castTo[ComplexFloat]
      def toComplexDouble: Tensor[ComplexDouble] = tensor.castTo[ComplexDouble]
      def toByte: Tensor[Byte] = tensor.castTo[Byte]
      def toShort: Tensor[Short] = tensor.castTo[Short]
      def toInt: Tensor[Int] = tensor.castTo[Int]
      def toLong: Tensor[Long] = tensor.castTo[Long]
      def toUByte: Tensor[UByte] = tensor.castTo[UByte]
      def toUShort: Tensor[UShort] = tensor.castTo[UShort]
      def toUInt: Tensor[UInt] = tensor.castTo[UInt]
      def toULong: Tensor[ULong] = tensor.castTo[ULong]
      def toQByte: Tensor[QByte] = tensor.castTo[QByte]
      def toQShort: Tensor[QShort] = tensor.castTo[QShort]
      def toQInt: Tensor[QInt] = tensor.castTo[QInt]
      def toQUByte: Tensor[QUByte] = tensor.castTo[QUByte]
      def toQUShort: Tensor[QUShort] = tensor.castTo[QUShort]
      def toResource: Tensor[Resource] = tensor.castTo[Resource]
      def toVariant: Tensor[Variant] = tensor.castTo[Variant]
    }
  }
}
