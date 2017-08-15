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

import spire.math.{UByte, UShort}

/**
  * @author Emmanouil Antonios Platanios
  */
package object types {
  private[api] trait ScopedAPI extends DataType.ScopedAPI {
    type DataType = types.DataType

    val STRING    : DataType.Aux[String]  = types.STRING
    val BOOLEAN   : DataType.Aux[Boolean] = types.BOOLEAN
    val FLOAT16   : DataType.Aux[Float]   = types.FLOAT16
    val FLOAT32   : DataType.Aux[Float]   = types.FLOAT32
    val FLOAT64   : DataType.Aux[Double]  = types.FLOAT64
    val BFLOAT16  : DataType.Aux[Float]   = types.BFLOAT16
    val COMPLEX64 : DataType.Aux[Double]  = types.COMPLEX64
    val COMPLEX128: DataType.Aux[Double]  = types.COMPLEX128
    val INT8      : DataType.Aux[Byte]    = types.INT8
    val INT16     : DataType.Aux[Short]   = types.INT16
    val INT32     : DataType.Aux[Int]     = types.INT32
    val INT64     : DataType.Aux[Long]    = types.INT64
    val UINT8     : DataType.Aux[UByte]   = types.UINT8
    val UINT16    : DataType.Aux[UShort]  = types.UINT16
    val QINT8     : DataType.Aux[Byte]    = types.QINT8
    val QINT16    : DataType.Aux[Short]   = types.QINT16
    val QINT32    : DataType.Aux[Int]     = types.QINT32
    val QUINT8    : DataType.Aux[UByte]   = types.QUINT8
    val QUINT16   : DataType.Aux[UShort]  = types.QUINT16
    val RESOURCE  : DataType.Aux[Long]    = types.RESOURCE
  }
}
