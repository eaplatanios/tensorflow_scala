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

package org.platanios.tensorflow.api.core

import org.tensorflow.framework.DataType._

/**
  * @author Emmanouil Antonios Platanios
  */
package object types {
  // TODO: [TYPES] Add some useful functionality to the following types.

  case class Half(data: Short) extends AnyVal
  case class TruncatedHalf(data: Short) extends AnyVal
  case class ComplexFloat(real: Float, imaginary: Float)
  case class ComplexDouble(real: Double, imaginary: Double)
  case class UByte(data: Byte) extends AnyVal
  case class UShort(data: Short) extends AnyVal
  case class UInt(data: Int) extends AnyVal
  case class ULong(data: Long) extends AnyVal
  case class QByte(data: Byte) extends AnyVal
  case class QShort(data: Short) extends AnyVal
  case class QInt(data: Int) extends AnyVal
  case class QUByte(data: Byte) extends AnyVal
  case class QUShort(data: Short) extends AnyVal
  case class Resource(data: Long) extends AnyVal
  case class Variant(data: Long) extends AnyVal

  //region Data Type Instances

  val STRING    : DataType[String]        = DataType[String]("String", 7, None, DT_STRING)
  val BOOLEAN   : DataType[Boolean]       = DataType[Boolean]("Boolean", 10, Some(1), DT_BOOL)
  val FLOAT16   : DataType[Half]          = DataType[Half]("Half", 19, Some(2), DT_HALF)
  val FLOAT32   : DataType[Float]         = DataType[Float]("Float", 1, Some(4), DT_FLOAT)
  val FLOAT64   : DataType[Double]        = DataType[Double]("Double", 2, Some(8), DT_DOUBLE)
  val BFLOAT16  : DataType[TruncatedHalf] = DataType[TruncatedHalf]("TruncatedFloat", 14, Some(2), DT_BFLOAT16)
  val COMPLEX64 : DataType[ComplexFloat]  = DataType[ComplexFloat]("ComplexFloat", 8, Some(8), DT_COMPLEX64)
  val COMPLEX128: DataType[ComplexDouble] = DataType[ComplexDouble]("ComplexDouble", 18, Some(16), DT_COMPLEX128)
  val INT8      : DataType[Byte]          = DataType[Byte]("Byte", 6, Some(1), DT_INT8)
  val INT16     : DataType[Short]         = DataType[Short]("Short", 5, Some(2), DT_INT16)
  val INT32     : DataType[Int]           = DataType[Int]("Int", 3, Some(4), DT_INT32)
  val INT64     : DataType[Long]          = DataType[Long]("Long", 9, Some(8), DT_INT64)
  val UINT8     : DataType[UByte]         = DataType[UByte]("UByte", 4, Some(1), DT_UINT8)
  val UINT16    : DataType[UShort]        = DataType[UShort]("UShort", 17, Some(2), DT_UINT16)
  val UINT32    : DataType[UInt]          = DataType[UInt]("UInt", 22, Some(4), DT_UINT32)
  val UINT64    : DataType[ULong]         = DataType[ULong]("ULong", 23, Some(8), DT_UINT64)
  val QINT8     : DataType[QByte]         = DataType[QByte]("QByte", 11, Some(1), DT_QINT8)
  val QINT16    : DataType[QShort]        = DataType[QShort]("QShort", 15, Some(2), DT_QINT16)
  val QINT32    : DataType[QInt]          = DataType[QInt]("QInt", 13, Some(4), DT_QINT32)
  val QUINT8    : DataType[QUByte]        = DataType[QUByte]("QUByte", 12, Some(1), DT_QUINT8)
  val QUINT16   : DataType[QUShort]       = DataType[QUShort]("QUShort", 16, Some(2), DT_QUINT16)
  val RESOURCE  : DataType[Resource]      = DataType[Resource]("Resource", 20, Some(1), DT_RESOURCE)
  val VARIANT   : DataType[Variant]       = DataType[Variant]("Variant", 21, Some(1), DT_VARIANT)

  //endregion Data Type Instances

  //region Type Traits

  trait TF[T] {
    @inline def dataType: org.platanios.tensorflow.api.core.types.DataType[T]
  }

  object TF extends TFLowPriority {
    def apply[T: TF]: TF[T] = {
      implicitly[TF[T]]
    }

    def fromDataType[T](dataType: org.platanios.tensorflow.api.core.types.DataType[T]): TF[T] = {
      val providedDataType = dataType
      new TF[T] {
        override def dataType: org.platanios.tensorflow.api.core.types.DataType[T] = {
          providedDataType
        }
      }
    }

    implicit val stringEvTF : TF[String]  = fromDataType(STRING)
    implicit val booleanEvTF: TF[Boolean] = fromDataType(BOOLEAN)
    implicit val floatEvTF  : TF[Float]   = fromDataType(FLOAT32)
    implicit val intEvTF    : TF[Int]     = fromDataType(INT32)
    implicit val longEvTF   : TF[Long]    = fromDataType(INT64)
  }

  trait TFLowPriority extends TFLowestPriority {
    implicit val doubleEvTF: TF[Double] = TF.fromDataType(FLOAT64)
    implicit val byteEvTF  : TF[Byte]   = TF.fromDataType(INT8)
    implicit val shortEvTF : TF[Short]  = TF.fromDataType(INT16)
  }

  trait TFLowestPriority {
    implicit val halfEvTF         : TF[Half]          = TF.fromDataType(FLOAT16)
    implicit val truncatedHalfEvTF: TF[TruncatedHalf] = TF.fromDataType(BFLOAT16)
    implicit val complexFloatEvTF : TF[ComplexFloat]  = TF.fromDataType(COMPLEX64)
    implicit val complexDoubleEvTF: TF[ComplexDouble] = TF.fromDataType(COMPLEX128)
    implicit val uByteEvTF        : TF[UByte]         = TF.fromDataType(UINT8)
    implicit val uShortEvTF       : TF[UShort]        = TF.fromDataType(UINT16)
    implicit val uIntEvTF         : TF[UInt]          = TF.fromDataType(UINT32)
    implicit val uLongEvTF        : TF[ULong]         = TF.fromDataType(UINT64)
    implicit val qByteEvTF        : TF[QByte]         = TF.fromDataType(QINT8)
    implicit val qShortEvTF       : TF[QShort]        = TF.fromDataType(QINT16)
    implicit val qIntEvTF         : TF[QInt]          = TF.fromDataType(QINT32)
    implicit val qUByteEvTF       : TF[QUByte]        = TF.fromDataType(QUINT8)
    implicit val qUShortEvTF      : TF[QUShort]       = TF.fromDataType(QUINT16)
    implicit val resourceEvTF     : TF[Resource]      = TF.fromDataType(RESOURCE)
    implicit val variantEvTF      : TF[Variant]       = TF.fromDataType(VARIANT)
  }

  import UnionTypes._

  type Float32OrFloat64 = Union[Float]#or[Double]
  type Float16OrFloat32OrFloat64 = Union[Half]#or[Float]#or[Double]
  type BFloat16OrFloat32OrFloat64 = Union[TruncatedHalf]#or[Float]#or[Double]
  type BFloat16OrFloat16OrFloat32 = Union[TruncatedHalf]#or[Half]#or[Float]
  type Decimal = Union[TruncatedHalf]#or[Half]#or[Float]#or[Double]
  type Int32OrInt64 = Union[Int]#or[Long]
  type Int32OrInt64OrFloat32OrFloat64 = Union[Int]#or[Long]#or[Float]#or[Double]
  type Int32OrInt64OrFloat16OrFloat32OrFloat64 = Union[Int]#or[Long]#or[Half]#or[Float]#or[Double]
  type Int32OrInt64OrUInt8 = Union[Int]#or[Long]#or[UByte]
  type SignedInteger = Union[Byte]#or[Short]#or[Int]#or[Long]
  type UnsignedInteger = Union[UByte]#or[UShort]#or[UInt]#or[ULong]
  type Integer = Union[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]
  type StringOrInteger = Union[String]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]
  type Real = Union[TruncatedHalf]#or[Half]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]
  type Complex = Union[ComplexFloat]#or[ComplexDouble]
  type NotQuantized = Union[TruncatedHalf]#or[Half]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[ComplexFloat]#or[ComplexDouble]
  type Quantized = Union[QByte]#or[QShort]#or[QInt]#or[QUByte]#or[QUShort]
  type Numeric = Union[TruncatedHalf]#or[Half]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[ComplexFloat]#or[ComplexDouble]#or[QByte]#or[QShort]#or[QInt]#or[QUByte]#or[QUShort]
  type BooleanOrNumeric = Union[Boolean]#or[Half]#or[Float]#or[Double]#or[Byte]#or[Short]#or[Int]#or[Long]#or[UByte]#or[UShort]#or[UInt]#or[ULong]#or[ComplexFloat]#or[ComplexDouble]#or[QByte]#or[QShort]#or[QInt]#or[QUByte]#or[QUShort]

  type IsFloat32OrFloat64[T] = Contains[T, Float32OrFloat64]
  type IsFloat16OrFloat32OrFloat64[T] = Contains[T, Float16OrFloat32OrFloat64]
  type IsBFloat16OrFloat32OrFloat64[T] = Contains[T, BFloat16OrFloat32OrFloat64]
  type IsBFloat16OrFloat16OrFloat32[T] = Contains[T, BFloat16OrFloat16OrFloat32]
  type IsDecimal[T] = Contains[T, Decimal]
  type IsInt32OrInt64[T] = Contains[T, Int32OrInt64]
  type IsInt32OrInt64OrFloat32OrFloat64[T] = Contains[T, Int32OrInt64OrFloat32OrFloat64]
  type IsInt32OrInt64OrFloat16OrFloat32OrFloat64[T] = Contains[T, Int32OrInt64OrFloat16OrFloat32OrFloat64]
  type IsInt32OrInt64OrUInt8[T] = Contains[T, Int32OrInt64OrUInt8]
  type IsIntOrUInt[T] = Contains[T, Integer]
  type IsStringOrIntOrUInt[T] = Contains[T, StringOrInteger]
  type IsReal[T] = Contains[T, Real]
  type IsComplex[T] = Contains[T, Complex]
  type IsNotQuantized[T] = Contains[T, NotQuantized]
  type IsQuantized[T] = Contains[T, Quantized]
  type IsNumeric[T] = Contains[T, Numeric]
  type IsBooleanOrNumeric[T] = Contains[T, BooleanOrNumeric]

  object IsFloat32OrFloat64 {
    def apply[T: IsFloat32OrFloat64]: IsFloat32OrFloat64[T] = implicitly[IsFloat32OrFloat64[T]]
  }

  object IsFloat16OrFloat32OrFloat64 {
    def apply[T: IsFloat16OrFloat32OrFloat64]: IsFloat16OrFloat32OrFloat64[T] = implicitly[IsFloat16OrFloat32OrFloat64[T]]
  }

  object IsBFloat16OrFloat32OrFloat64 {
    def apply[T: IsBFloat16OrFloat32OrFloat64]: IsBFloat16OrFloat32OrFloat64[T] = implicitly[IsBFloat16OrFloat32OrFloat64[T]]
  }

  object IsBFloat16OrFloat16OrFloat32 {
    def apply[T: IsBFloat16OrFloat16OrFloat32]: IsBFloat16OrFloat16OrFloat32[T] = implicitly[IsBFloat16OrFloat16OrFloat32[T]]
  }

  object IsDecimal {
    def apply[T: IsDecimal]: IsDecimal[T] = implicitly[IsDecimal[T]]
  }

  object IsInt32OrInt64 {
    def apply[T: IsInt32OrInt64]: IsInt32OrInt64[T] = implicitly[IsInt32OrInt64[T]]
  }

  object IsInt32OrInt64OrFloat32OrFloat64 {
    def apply[T: IsInt32OrInt64OrFloat32OrFloat64]: IsInt32OrInt64OrFloat32OrFloat64[T] = implicitly[IsInt32OrInt64OrFloat32OrFloat64[T]]
  }

  object IsInt32OrInt64OrFloat16OrFloat32OrFloat64 {
    def apply[T: IsInt32OrInt64OrFloat16OrFloat32OrFloat64]: IsInt32OrInt64OrFloat16OrFloat32OrFloat64[T] = implicitly[IsInt32OrInt64OrFloat16OrFloat32OrFloat64[T]]
  }

  object IsInt32OrInt64OrUInt8 {
    def apply[T: IsInt32OrInt64OrUInt8]: IsInt32OrInt64OrUInt8[T] = implicitly[IsInt32OrInt64OrUInt8[T]]
  }

  object IsIntOrUInt {
    def apply[T: IsIntOrUInt]: IsIntOrUInt[T] = implicitly[IsIntOrUInt[T]]
  }

  object IsStringOrIntOrUInt {
    def apply[T: IsStringOrIntOrUInt]: IsStringOrIntOrUInt[T] = implicitly[IsStringOrIntOrUInt[T]]
  }

  object IsReal {
    def apply[T: IsReal]: IsReal[T] = implicitly[IsReal[T]]
  }

  object IsComplex {
    def apply[T: IsComplex]: IsComplex[T] = implicitly[IsComplex[T]]
  }

  object IsNotQuantized {
    def apply[T: IsNotQuantized]: IsNotQuantized[T] = implicitly[IsNotQuantized[T]]
  }

  object IsQuantized {
    def apply[T: IsQuantized]: IsQuantized[T] = implicitly[IsQuantized[T]]
  }

  object IsNumeric {
    def apply[T: IsNumeric]: IsNumeric[T] = implicitly[IsNumeric[T]]
  }

  object IsBooleanOrNumeric {
    def apply[T: IsBooleanOrNumeric]: IsBooleanOrNumeric[T] = implicitly[IsBooleanOrNumeric[T]]
  }
}
