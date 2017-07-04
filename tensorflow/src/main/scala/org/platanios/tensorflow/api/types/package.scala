/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api

/**
  * @author Emmanouil Antonios Platanios
  */
package object types {
  private[api] trait API {
    type SupportedType[T] = types.SupportedType[T]
    type FixedSizeSupportedType[T] = types.FixedSizeSupportedType[T]
    type NumericSupportedType[T] = types.NumericSupportedType[T]
    type SignedNumericSupportedType[T] = types.SignedNumericSupportedType[T]
    type RealNumericSupportedType[T] = types.RealNumericSupportedType[T]
    type ComplexNumericSupportedType[T] = types.ComplexNumericSupportedType[T]

    type DataType = types.DataType
    type FixedSizeDataType = types.FixedSizeDataType
    type NumericDataType = types.NumericDataType
    type SignedNumericDataType = types.SignedNumericDataType
    type RealNumericDataType = types.RealNumericDataType
    type ComplexNumericDataType = types.ComplexNumericDataType

    val DataType = types.DataType

    val STRING  : STRING   = types.STRING
    val BOOLEAN : BOOLEAN  = types.BOOLEAN
    // val FLOAT16 : FLOAT16 = types.TFFloat16
    val FLOAT32 : FLOAT32  = types.FLOAT32
    val FLOAT64 : FLOAT64  = types.FLOAT64
    // val BFLOAT16 : BFLOAT16 = types.TFBFloat16
    // val COMPLEX64 : COMPLEX64 = types.TFComplex64
    // val COMPLEX128 : COMPLEX128 = types.TFComplex128
    val INT8    : INT8     = types.INT8
    val INT16   : INT16    = types.INT16
    val INT32   : INT32    = types.INT32
    val INT64   : INT64    = types.INT64
    val UINT8   : UINT8    = types.UINT8
    val UINT16  : UINT16   = types.UINT16
    val QINT8   : QINT8    = types.QINT8
    val QINT16  : QINT16   = types.QINT16
    val QINT32  : QINT32   = types.QINT32
    val QUINT8  : QUINT8   = types.QUINT8
    val QUINT16 : QUINT16  = types.QUINT16
    val RESOURCE: RESOURCE = types.RESOURCE
  }

  type STRING   = types.STRING.type
  type BOOLEAN  = types.BOOLEAN.type
  // type FLOAT16 = types.TFFloat16.type
  type FLOAT32  = types.FLOAT32.type
  type FLOAT64  = types.FLOAT64.type
  // type BFLOAT16 = types.TFBFloat16.type
  // type COMPLEX64 = types.TFComplex64.type
  // type COMPLEX128 = types.TFComplex128.type
  type INT8     = types.INT8.type
  type INT16    = types.INT16.type
  type INT32    = types.INT32.type
  type INT64    = types.INT64.type
  type UINT8    = types.UINT8.type
  type UINT16   = types.UINT16.type
  type QINT8    = types.QINT8.type
  type QINT16   = types.QINT16.type
  type QINT32   = types.QINT32.type
  type QUINT8   = types.QUINT8.type
  type QUINT16  = types.QUINT16.type
  type RESOURCE = types.RESOURCE.type

}
