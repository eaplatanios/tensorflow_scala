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

  object AnyDefault {
    def apply[T: AnyDefault]: AnyDefault[T] = implicitly[AnyDefault[T]]
  }

  object StringDefault {
    def apply[T: StringDefault]: StringDefault[T] = implicitly[StringDefault[T]]
  }

  object BooleanDefault {
    def apply[T: BooleanDefault]: BooleanDefault[T] = implicitly[BooleanDefault[T]]
  }

  object HalfDefault {
    def apply[T: HalfDefault]: HalfDefault[T] = implicitly[HalfDefault[T]]
  }

  object FloatDefault {
    def apply[T: FloatDefault]: FloatDefault[T] = implicitly[FloatDefault[T]]
  }

  object DoubleDefault {
    def apply[T: DoubleDefault]: DoubleDefault[T] = implicitly[DoubleDefault[T]]
  }

  object TruncatedHalfDefault {
    def apply[T: TruncatedHalfDefault]: TruncatedHalfDefault[T] = implicitly[TruncatedHalfDefault[T]]
  }

  object ComplexFloatDefault {
    def apply[T: ComplexFloatDefault]: ComplexFloatDefault[T] = implicitly[ComplexFloatDefault[T]]
  }

  object ComplexDoubleDefault {
    def apply[T: ComplexDoubleDefault]: ComplexDoubleDefault[T] = implicitly[ComplexDoubleDefault[T]]
  }

  object ByteDefault {
    def apply[T: ByteDefault]: ByteDefault[T] = implicitly[ByteDefault[T]]
  }

  object ShortDefault {
    def apply[T: ShortDefault]: ShortDefault[T] = implicitly[ShortDefault[T]]
  }

  object IntDefault {
    def apply[T: IntDefault]: IntDefault[T] = implicitly[IntDefault[T]]
  }

  object LongDefault {
    def apply[T: LongDefault]: LongDefault[T] = implicitly[LongDefault[T]]
  }

  object UByteDefault {
    def apply[T: UByteDefault]: UByteDefault[T] = implicitly[UByteDefault[T]]
  }

  object UShortDefault {
    def apply[T: UShortDefault]: UShortDefault[T] = implicitly[UShortDefault[T]]
  }

  object UIntDefault {
    def apply[T: UIntDefault]: UIntDefault[T] = implicitly[UIntDefault[T]]
  }

  object ULongDefault {
    def apply[T: ULongDefault]: ULongDefault[T] = implicitly[ULongDefault[T]]
  }

  object QByteDefault {
    def apply[T: QByteDefault]: QByteDefault[T] = implicitly[QByteDefault[T]]
  }

  object QShortDefault {
    def apply[T: QShortDefault]: QShortDefault[T] = implicitly[QShortDefault[T]]
  }

  object QIntDefault {
    def apply[T: QIntDefault]: QIntDefault[T] = implicitly[QIntDefault[T]]
  }

  object QUByteDefault {
    def apply[T: QUByteDefault]: QUByteDefault[T] = implicitly[QUByteDefault[T]]
  }

  object QUShortDefault {
    def apply[T: QUShortDefault]: QUShortDefault[T] = implicitly[QUShortDefault[T]]
  }

  object ResourceDefault {
    def apply[T: ResourceDefault]: ResourceDefault[T] = implicitly[ResourceDefault[T]]
  }

  object VariantDefault {
    def apply[T: VariantDefault]: VariantDefault[T] = implicitly[VariantDefault[T]]
  }
}
