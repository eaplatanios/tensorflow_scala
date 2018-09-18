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

package org.platanios.tensorflow.api.types

/**
  * @author Emmanouil Antonios Platanios
  */
trait AllowedCast[TARGET] {
  type S
  val targetDataType: DataType[TARGET]
}

object AllowedCast {
  type Aux[SOURCE, TARGET] = AllowedCast[TARGET] {
    type S = SOURCE
  }

  implicit val int8ToFloat32: AllowedCast.Aux[Byte, Float]  = new ToFloat[Byte]
  implicit val int8ToFloat64: AllowedCast.Aux[Byte, Double] = new ToFloat64[Byte]
  implicit val int8ToInt16  : AllowedCast.Aux[Byte, Short]  = new ToInt16[Byte]
  implicit val int8ToInt32  : AllowedCast.Aux[Byte, Int]    = new ToInt32[Byte]
  implicit val int8ToInt64  : AllowedCast.Aux[Byte, Long]   = new ToInt64[Byte]

  implicit val int16ToFloat32: AllowedCast.Aux[Short, Float]  = new ToFloat[Short]
  implicit val int16ToFloat64: AllowedCast.Aux[Short, Double] = new ToFloat64[Short]
  implicit val int16ToInt32  : AllowedCast.Aux[Short, Int]    = new ToInt32[Short]
  implicit val int16ToInt64  : AllowedCast.Aux[Short, Long]   = new ToInt64[Short]

  implicit val int32ToFloat32: AllowedCast.Aux[Int, Float]  = new ToFloat[Int]
  implicit val int32ToFloat64: AllowedCast.Aux[Int, Double] = new ToFloat64[Int]
  implicit val int32ToInt64  : AllowedCast.Aux[Int, Long]   = new ToInt64[Int]

  implicit val int64ToFloat32: AllowedCast.Aux[Long, Float]  = new ToFloat[Long]
  implicit val int64ToFloat64: AllowedCast.Aux[Long, Double] = new ToFloat64[Long]
  implicit val int64ToInt64  : AllowedCast.Aux[Long, Long]   = new ToInt64[Long]

  implicit val uint8ToFloat32: AllowedCast.Aux[UByte, Float]  = new ToFloat[UByte]
  implicit val uint8ToFloat64: AllowedCast.Aux[UByte, Double] = new ToFloat64[UByte]
  implicit val uint8ToInt16  : AllowedCast.Aux[UByte, Short]  = new ToInt16[UByte]
  implicit val uint8ToInt32  : AllowedCast.Aux[UByte, Int]    = new ToInt32[UByte]
  implicit val uint8ToInt64  : AllowedCast.Aux[UByte, Long]   = new ToInt64[UByte]
  implicit val uint8ToUInt16 : AllowedCast.Aux[UByte, UShort] = new ToUInt16[UByte]
  implicit val uint8ToUInt32 : AllowedCast.Aux[UByte, UInt]   = new ToUInt32[UByte]
  implicit val uint8ToUInt64 : AllowedCast.Aux[UByte, ULong]  = new ToUInt64[UByte]

  implicit val uint16ToFloat32: AllowedCast.Aux[UShort, Float]  = new ToFloat[UShort]
  implicit val uint16ToFloat64: AllowedCast.Aux[UShort, Double] = new ToFloat64[UShort]
  implicit val uint16ToInt32  : AllowedCast.Aux[UShort, Int]    = new ToInt32[UShort]
  implicit val uint16ToInt64  : AllowedCast.Aux[UShort, Long]   = new ToInt64[UShort]
  implicit val uint16ToUInt32 : AllowedCast.Aux[UShort, UInt]   = new ToUInt32[UShort]
  implicit val uint16ToUInt64 : AllowedCast.Aux[UShort, ULong]  = new ToUInt64[UShort]

  implicit val uint32ToFloat32: AllowedCast.Aux[UInt, Float]  = new ToFloat[UInt]
  implicit val uint32ToFloat64: AllowedCast.Aux[UInt, Double] = new ToFloat64[UInt]
  implicit val uint32ToInt64  : AllowedCast.Aux[UInt, Long]   = new ToInt64[UInt]
  implicit val uint32ToUInt64 : AllowedCast.Aux[UInt, ULong]  = new ToUInt64[UInt]

  implicit val uint64ToFloat32: AllowedCast.Aux[ULong, Float]  = new ToFloat[ULong]
  implicit val uint64ToFloat64: AllowedCast.Aux[ULong, Double] = new ToFloat64[ULong]
  implicit val uint64ToInt64  : AllowedCast.Aux[ULong, Long]   = new ToInt64[ULong]
  implicit val uint64ToUInt64 : AllowedCast.Aux[ULong, ULong]  = new ToUInt64[ULong]

  private[types] class ToString[SOURCE] extends AllowedCast[String] {
    override type S = SOURCE
    val targetDataType: DataType[String] = STRING
  }

  private[types] class ToBoolean[SOURCE] extends AllowedCast[Boolean] {
    override type S = SOURCE
    val targetDataType: DataType[Boolean] = BOOLEAN
  }

  private[types] class ToFloat16[SOURCE] extends AllowedCast[Half] {
    override type S = SOURCE
    val targetDataType: DataType[Half] = FLOAT16
  }

  private[types] class ToFloat[SOURCE] extends AllowedCast[Float] {
    override type S = SOURCE
    val targetDataType: DataType[Float] = FLOAT32
  }

  private[types] class ToFloat64[SOURCE] extends AllowedCast[Double] {
    override type S = SOURCE
    val targetDataType: DataType[Double] = FLOAT64
  }

  private[types] class ToBFloat16[SOURCE] extends AllowedCast[TruncatedHalf] {
    override type S = SOURCE
    val targetDataType: DataType[TruncatedHalf] = BFLOAT16
  }

  private[types] class ToComplex64[SOURCE] extends AllowedCast[ComplexFloat] {
    override type S = SOURCE
    val targetDataType: DataType[ComplexFloat] = COMPLEX64
  }

  private[types] class ToComplex128[SOURCE] extends AllowedCast[ComplexDouble] {
    override type S = SOURCE
    val targetDataType: DataType[ComplexDouble] = COMPLEX128
  }

  private[types] class ToInt8[SOURCE] extends AllowedCast[Byte] {
    override type S = SOURCE
    override val targetDataType: DataType[Byte] = INT8
  }

  private[types] class ToInt16[SOURCE] extends AllowedCast[Short] {
    override type S = SOURCE
    override val targetDataType: DataType[Short] = INT16
  }

  private[types] class ToInt32[SOURCE] extends AllowedCast[Int] {
    override type S = SOURCE
    override val targetDataType: DataType[Int] = INT32
  }

  private[types] class ToInt64[SOURCE] extends AllowedCast[Long] {
    override type S = SOURCE
    override val targetDataType: DataType[Long] = INT64
  }

  private[types] class ToUInt8[SOURCE] extends AllowedCast[UByte] {
    override type S = SOURCE
    override val targetDataType: DataType[UByte] = UINT8
  }

  private[types] class ToUInt16[SOURCE] extends AllowedCast[UShort] {
    override type S = SOURCE
    override val targetDataType: DataType[UShort] = UINT16
  }

  private[types] class ToUInt32[SOURCE] extends AllowedCast[UInt] {
    override type S = SOURCE
    override val targetDataType: DataType[UInt] = UINT32
  }

  private[types] class ToUInt64[SOURCE] extends AllowedCast[ULong] {
    override type S = SOURCE
    override val targetDataType: DataType[ULong] = UINT64
  }

  private[types] class ToQInt8[SOURCE] extends AllowedCast[QByte] {
    override type S = SOURCE
    override val targetDataType: DataType[QByte] = QINT8
  }

  private[types] class ToQInt16[SOURCE] extends AllowedCast[QShort] {
    override type S = SOURCE
    override val targetDataType: DataType[QShort] = QINT16
  }

  private[types] class ToQInt32[SOURCE] extends AllowedCast[QInt] {
    override type S = SOURCE
    override val targetDataType: DataType[QInt] = QINT32
  }

  private[types] class ToQUInt8[SOURCE] extends AllowedCast[QUByte] {
    override type S = SOURCE
    override val targetDataType: DataType[QUByte] = QUINT8
  }

  private[types] class ToQUInt16[SOURCE] extends AllowedCast[QUShort] {
    override type S = SOURCE
    override val targetDataType: DataType[QUShort] = QUINT16
  }
}
