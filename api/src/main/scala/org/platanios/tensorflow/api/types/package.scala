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

package org.platanios.tensorflow.api

/**
  * @author Emmanouil Antonios Platanios
  */
package object types {
  private[api] trait API extends DataType.API

  type STRING = types.STRING.type
  type BOOLEAN = types.BOOLEAN.type
  type FLOAT16 = types.FLOAT16.type
  type FLOAT32 = types.FLOAT32.type
  type FLOAT64 = types.FLOAT64.type
  type BFLOAT16 = types.BFLOAT16.type
  type COMPLEX64 = types.COMPLEX64.type
  type COMPLEX128 = types.COMPLEX128.type
  type INT8 = types.INT8.type
  type INT16 = types.INT16.type
  type INT32 = types.INT32.type
  type INT64 = types.INT64.type
  type UINT8 = types.UINT8.type
  type UINT16 = types.UINT16.type
  type UINT32 = types.UINT32.type
  type UINT64 = types.UINT64.type
  type QINT8 = types.QINT8.type
  type QINT16 = types.QINT16.type
  type QINT32 = types.QINT32.type
  type QUINT8 = types.QUINT8.type
  type QUINT16 = types.QUINT16.type
  type RESOURCE = types.RESOURCE.type
  type VARIANT = types.VARIANT.type
}
