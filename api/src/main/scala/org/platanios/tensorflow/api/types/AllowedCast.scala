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
trait AllowedCast[TARGET <: DataType] {
  type S <: DataType
  val targetDataType: TARGET
}

object AllowedCast {
  implicit val int8ToFloat32: AllowedCast.Aux[INT8, FLOAT32] = new ToFloat32[INT8]
  implicit val int8ToFloat64: AllowedCast.Aux[INT8, FLOAT64] = new ToFloat64[INT8]
  implicit val int8ToInt16  : AllowedCast.Aux[INT8, INT16]   = new ToInt16[INT8]
  implicit val int8ToInt32  : AllowedCast.Aux[INT8, INT32]   = new ToInt32[INT8]
  implicit val int8ToInt64  : AllowedCast.Aux[INT8, INT64]   = new ToInt64[INT8]

  implicit val int16ToFloat32: AllowedCast.Aux[INT16, FLOAT32] = new ToFloat32[INT16]
  implicit val int16ToFloat64: AllowedCast.Aux[INT16, FLOAT64] = new ToFloat64[INT16]
  implicit val int16ToInt32  : AllowedCast.Aux[INT16, INT32]   = new ToInt32[INT16]
  implicit val int16ToInt64  : AllowedCast.Aux[INT16, INT64]   = new ToInt64[INT16]

  implicit val int32ToFloat32: AllowedCast.Aux[INT32, FLOAT32] = new ToFloat32[INT32]
  implicit val int32ToFloat64: AllowedCast.Aux[INT32, FLOAT64] = new ToFloat64[INT32]
  implicit val int32ToInt64  : AllowedCast.Aux[INT32, INT64]   = new ToInt64[INT32]

  implicit val int64ToFloat32: AllowedCast.Aux[INT64, FLOAT32] = new ToFloat32[INT64]
  implicit val int64ToFloat64: AllowedCast.Aux[INT64, FLOAT64] = new ToFloat64[INT64]
  implicit val int64ToInt64  : AllowedCast.Aux[INT64, INT64]   = new ToInt64[INT64]

  implicit val uint8ToFloat32: AllowedCast.Aux[UINT8, FLOAT32] = new ToFloat32[UINT8]
  implicit val uint8ToFloat64: AllowedCast.Aux[UINT8, FLOAT64] = new ToFloat64[UINT8]
  implicit val uint8ToInt16  : AllowedCast.Aux[UINT8, INT16]   = new ToInt16[UINT8]
  implicit val uint8ToInt32  : AllowedCast.Aux[UINT8, INT32]   = new ToInt32[UINT8]
  implicit val uint8ToInt64  : AllowedCast.Aux[UINT8, INT64]   = new ToInt64[UINT8]
  implicit val uint8ToUInt16 : AllowedCast.Aux[UINT8, UINT16]  = new ToUInt16[UINT8]
  implicit val uint8ToUInt32 : AllowedCast.Aux[UINT8, UINT32]  = new ToUInt32[UINT8]
  implicit val uint8ToUInt64 : AllowedCast.Aux[UINT8, UINT64]  = new ToUInt64[UINT8]

  implicit val uint16ToFloat32: AllowedCast.Aux[UINT16, FLOAT32] = new ToFloat32[UINT16]
  implicit val uint16ToFloat64: AllowedCast.Aux[UINT16, FLOAT64] = new ToFloat64[UINT16]
  implicit val uint16ToInt32  : AllowedCast.Aux[UINT16, INT32]   = new ToInt32[UINT16]
  implicit val uint16ToInt64  : AllowedCast.Aux[UINT16, INT64]   = new ToInt64[UINT16]
  implicit val uint16ToUInt32 : AllowedCast.Aux[UINT16, UINT32]  = new ToUInt32[UINT16]
  implicit val uint16ToUInt64 : AllowedCast.Aux[UINT16, UINT64]  = new ToUInt64[UINT16]

  implicit val uint32ToFloat32: AllowedCast.Aux[UINT32, FLOAT32] = new ToFloat32[UINT32]
  implicit val uint32ToFloat64: AllowedCast.Aux[UINT32, FLOAT64] = new ToFloat64[UINT32]
  implicit val uint32ToInt64  : AllowedCast.Aux[UINT32, INT64]   = new ToInt64[UINT32]
  implicit val uint32ToUInt64 : AllowedCast.Aux[UINT32, UINT64]  = new ToUInt64[UINT32]

  implicit val uint64ToFloat32: AllowedCast.Aux[UINT64, FLOAT32] = new ToFloat32[UINT64]
  implicit val uint64ToFloat64: AllowedCast.Aux[UINT64, FLOAT64] = new ToFloat64[UINT64]
  implicit val uint64ToInt64  : AllowedCast.Aux[UINT64, INT64]   = new ToInt64[UINT64]
  implicit val uint64ToUInt64 : AllowedCast.Aux[UINT64, UINT64]  = new ToUInt64[UINT64]

  type Aux[SOURCE <: DataType, TARGET <: DataType] = AllowedCast[TARGET] {
    type S = SOURCE
  }

  private[types] class ToString[SOURCE <: DataType] extends AllowedCast[STRING] {
    override type S = SOURCE
    val targetDataType: STRING = STRING
  }

  private[types] class ToBoolean[SOURCE <: DataType] extends AllowedCast[BOOLEAN] {
    override type S = SOURCE
    val targetDataType: BOOLEAN = BOOLEAN
  }

  private[types] class ToFloat16[SOURCE <: DataType] extends AllowedCast[FLOAT16] {
    override type S = SOURCE
    val targetDataType: FLOAT16 = FLOAT16
  }

  private[types] class ToFloat32[SOURCE <: DataType] extends AllowedCast[FLOAT32] {
    override type S = SOURCE
    val targetDataType: FLOAT32 = FLOAT32
  }

  private[types] class ToFloat64[SOURCE <: DataType] extends AllowedCast[FLOAT64] {
    override type S = SOURCE
    val targetDataType: FLOAT64 = FLOAT64
  }

  private[types] class ToBFloat16[SOURCE <: DataType] extends AllowedCast[BFLOAT16] {
    override type S = SOURCE
    val targetDataType: BFLOAT16 = BFLOAT16
  }

  private[types] class ToComplex64[SOURCE <: DataType] extends AllowedCast[COMPLEX64] {
    override type S = SOURCE
    val targetDataType: COMPLEX64 = COMPLEX64
  }

  private[types] class ToComplex128[SOURCE <: DataType] extends AllowedCast[COMPLEX128] {
    override type S = SOURCE
    val targetDataType: COMPLEX128 = COMPLEX128
  }

  private[types] class ToInt8[SOURCE <: DataType] extends AllowedCast[INT8] {
    override type S = SOURCE
    override val targetDataType: INT8 = INT8
  }

  private[types] class ToInt16[SOURCE <: DataType] extends AllowedCast[INT16] {
    override type S = SOURCE
    override val targetDataType: INT16 = INT16
  }

  private[types] class ToInt32[SOURCE <: DataType] extends AllowedCast[INT32] {
    override type S = SOURCE
    override val targetDataType: INT32 = INT32
  }

  private[types] class ToInt64[SOURCE <: DataType] extends AllowedCast[INT64] {
    override type S = SOURCE
    override val targetDataType: INT64 = INT64
  }

  private[types] class ToUInt8[SOURCE <: DataType] extends AllowedCast[UINT8] {
    override type S = SOURCE
    override val targetDataType: UINT8 = UINT8
  }

  private[types] class ToUInt16[SOURCE <: DataType] extends AllowedCast[UINT16] {
    override type S = SOURCE
    override val targetDataType: UINT16 = UINT16
  }

  private[types] class ToUInt32[SOURCE <: DataType] extends AllowedCast[UINT32] {
    override type S = SOURCE
    override val targetDataType: UINT32 = UINT32
  }

  private[types] class ToUInt64[SOURCE <: DataType] extends AllowedCast[UINT64] {
    override type S = SOURCE
    override val targetDataType: UINT64 = UINT64
  }

  private[types] class ToQInt8[SOURCE <: DataType] extends AllowedCast[QINT8] {
    override type S = SOURCE
    override val targetDataType: QINT8 = QINT8
  }

  private[types] class ToQInt16[SOURCE <: DataType] extends AllowedCast[QINT16] {
    override type S = SOURCE
    override val targetDataType: QINT16 = QINT16
  }

  private[types] class ToQInt32[SOURCE <: DataType] extends AllowedCast[QINT32] {
    override type S = SOURCE
    override val targetDataType: QINT32 = QINT32
  }

  private[types] class ToQUInt8[SOURCE <: DataType] extends AllowedCast[QUINT8] {
    override type S = SOURCE
    override val targetDataType: QUINT8 = QUINT8
  }

  private[types] class ToQUInt16[SOURCE <: DataType] extends AllowedCast[QUINT16] {
    override type S = SOURCE
    override val targetDataType: QUINT16 = QUINT16
  }

  private[types] class ToResource[SOURCE <: DataType] extends AllowedCast[RESOURCE] {
    override type S = SOURCE
    override val targetDataType: RESOURCE = RESOURCE
  }

  private[types] class ToVariant[SOURCE <: DataType] extends AllowedCast[VARIANT] {
    override type S = SOURCE
    override val targetDataType: VARIANT = VARIANT
  }
}
