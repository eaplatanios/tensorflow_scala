/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.core.types

import org.platanios.tensorflow.api.core.types.DataType._

/**
  * @author Emmanouil Antonios Platanios
  */
private[core] trait Implicits {
  // implicit def scalaStringToTFString(x: String.type): DataType[String] = STRING
  implicit def scalaBooleanToTFBoolean(x: Boolean.type): DataType[Boolean] = BOOLEAN
  implicit def scalaHalfToTFFloat16(x: Half.type): DataType[Half] = FLOAT16
  implicit def scalaFloatToTFFloat32(x: Float.type): DataType[Float] = FLOAT32
  implicit def scalaDoubleToTFFloat64(x: Double.type): DataType[Double] = FLOAT64
  implicit def scalaTruncatedHalfToTFBFloat16(x: TruncatedHalf.type): DataType[TruncatedHalf] = BFLOAT16
  implicit def scalaComplexFloatToTFComplex64(x: ComplexFloat.type): DataType[ComplexFloat] = COMPLEX64
  implicit def scalaComplexDoubleToTFComplex128(x: ComplexDouble.type): DataType[ComplexDouble] = COMPLEX128
  implicit def scalaByteToTFInt8(x: Byte.type): DataType[Byte] = INT8
  implicit def scalaShortToTFInt16(x: Short.type): DataType[Short] = INT16
  implicit def scalaIntToTFInt32(x: Int.type): DataType[Int] = INT32
  implicit def scalaLongToTFInt64(x: Long.type): DataType[Long] = INT64
  implicit def scalaUByteToTFUInt8(x: UByte.type): DataType[UByte] = UINT8
  implicit def scalaUShortToTFUInt16(x: UShort.type): DataType[UShort] = UINT16
  implicit def scalaUIntToTFUInt32(x: UInt.type): DataType[UInt] = UINT32
  implicit def scalaULongToTFUInt64(x: ULong.type): DataType[ULong] = UINT64
  implicit def scalaQByteToTFQInt8(x: QByte.type): DataType[QByte] = QINT8
  implicit def scalaQShortToTFQInt16(x: QShort.type): DataType[QShort] = QINT16
  implicit def scalaQIntToTFQInt32(x: QInt.type): DataType[QInt] = QINT32
  implicit def scalaQUByteToTFQUInt8(x: QUByte.type): DataType[QUByte] = QUINT8
  implicit def scalaQUShortToTFQUInt16(x: QUShort.type): DataType[QUShort] = QUINT16
  implicit def scalaResourceToTFResource(x: Resource.type): DataType[Resource] = RESOURCE
  implicit def scalaVariantToTFVariant(x: Variant.type): DataType[Variant] = VARIANT

  implicit def dataTypeAsUntyped[T](dataType: DataType[T]): DataType[Any] = {
    dataType.asInstanceOf[DataType[Any]]
  }

  implicit def dataTypeArrayAsUntyped(
      dataTypes: Array[DataType[_]]
  ): Array[DataType[Any]] = {
    dataTypes.map(_.asInstanceOf[DataType[Any]])
  }

  implicit def dataTypeSeqAsUntyped(
      dataTypes: Seq[DataType[_]]
  ): Seq[DataType[Any]] = {
    dataTypes.map(_.asInstanceOf[DataType[Any]])
  }
}
