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

    val STRING     = types.STRING
    val BOOLEAN    = types.BOOLEAN
    val FLOAT16    = types.FLOAT16
    val FLOAT32    = types.FLOAT32
    val FLOAT64    = types.FLOAT64
    val BFLOAT16   = types.BFLOAT16
    val COMPLEX64  = types.COMPLEX64
    val COMPLEX128 = types.COMPLEX128
    val INT8       = types.INT8
    val INT16      = types.INT16
    val INT32      = types.INT32
    val INT64      = types.INT64
    val UINT8      = types.UINT8
    val UINT16     = types.UINT16
    val QINT8      = types.QINT8
    val QINT16     = types.QINT16
    val QINT32     = types.QINT32
    val QUINT8     = types.QUINT8
    val QUINT16    = types.QUINT16
    val RESOURCE   = types.RESOURCE
  }

  private[api] object API extends API
}
