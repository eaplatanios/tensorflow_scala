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

package org.platanios.tensorflow.api.utilities

import org.platanios.tensorflow.api.core.types._

/**
  * @author Emmanouil Antonios Platanios
  */
trait DefaultsTo[Type, Default]

object DefaultsTo {
  implicit def defaultDefaultsTo[T]: DefaultsTo[T, T] = null
  implicit def fallback[T, D]: DefaultsTo[T, D] = null

  type AnyDefault[T] = DefaultsTo[T, Any]
  type StringDefault[T] = DefaultsTo[T, String]
  type BooleanDefault[T] = DefaultsTo[T, Boolean]
  type HalfDefault[T] = DefaultsTo[T, Half]
  type FloatDefault[T] = DefaultsTo[T, Float]
  type DoubleDefault[T] = DefaultsTo[T, Double]
  type TruncatedHalfDefault[T] = DefaultsTo[T, TruncatedHalf]
  type ComplexFloatDefault[T] = DefaultsTo[T, ComplexFloat]
  type ComplexDoubleDefault[T] = DefaultsTo[T, ComplexDouble]
  type ByteDefault[T] = DefaultsTo[T, Byte]
  type ShortDefault[T] = DefaultsTo[T, Short]
  type IntDefault[T] = DefaultsTo[T, Int]
  type LongDefault[T] = DefaultsTo[T, Long]
  type UByteDefault[T] = DefaultsTo[T, UByte]
  type UShortDefault[T] = DefaultsTo[T, UShort]
  type UIntDefault[T] = DefaultsTo[T, UInt]
  type ULongDefault[T] = DefaultsTo[T, ULong]
  type QByteDefault[T] = DefaultsTo[T, QByte]
  type QShortDefault[T] = DefaultsTo[T, QShort]
  type QIntDefault[T] = DefaultsTo[T, QInt]
  type QUByteDefault[T] = DefaultsTo[T, QUByte]
  type QUShortDefault[T] = DefaultsTo[T, QUShort]
  type ResourceDefault[T] = DefaultsTo[T, Resource]
  type VariantDefault[T] = DefaultsTo[T, Variant]
}
