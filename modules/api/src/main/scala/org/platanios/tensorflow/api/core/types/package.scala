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

  case class Half(private[types] val data: Short) extends AnyVal
  case class TruncatedHalf(private[types] val data: Short) extends AnyVal
  case class ComplexFloat(real: Float, imaginary: Float)
  case class ComplexDouble(real: Double, imaginary: Double)
  case class UByte(private[types] val data: Byte) extends AnyVal
  case class UShort(private[types] val data: Short) extends AnyVal
  case class UInt(private[types] val data: Int) extends AnyVal
  case class ULong(private[types] val data: Long) extends AnyVal
  case class QByte(private[types] val data: Byte) extends AnyVal
  case class QShort(private[types] val data: Short) extends AnyVal
  case class QInt(private[types] val data: Int) extends AnyVal
  case class QUByte(private[types] val data: Byte) extends AnyVal
  case class QUShort(private[types] val data: Short) extends AnyVal
  case class Resource(private[types] val data: Long) extends AnyVal
  case class Variant(private[types] val data: Long) extends AnyVal

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

  object TF {
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

    implicit val stringEvTF       : TF[String]        = fromDataType(STRING)
    implicit val booleanEvTF      : TF[Boolean]       = fromDataType(BOOLEAN)
    implicit val halfEvTF         : TF[Half]          = fromDataType(FLOAT16)
    implicit val floatEvTF        : TF[Float]         = fromDataType(FLOAT32)
    implicit val doubleEvTF       : TF[Double]        = fromDataType(FLOAT64)
    implicit val truncatedHalfEvTF: TF[TruncatedHalf] = fromDataType(BFLOAT16)
    implicit val complexFloatEvTF : TF[ComplexFloat]  = fromDataType(COMPLEX64)
    implicit val complexDoubleEvTF: TF[ComplexDouble] = fromDataType(COMPLEX128)
    implicit val byteEvTF         : TF[Byte]          = fromDataType(INT8)
    implicit val shortEvTF        : TF[Short]         = fromDataType(INT16)
    implicit val intEvTF          : TF[Int]           = fromDataType(INT32)
    implicit val longEvTF         : TF[Long]          = fromDataType(INT64)
    implicit val uByteEvTF        : TF[UByte]         = fromDataType(UINT8)
    implicit val uShortEvTF       : TF[UShort]        = fromDataType(UINT16)
    implicit val uIntEvTF         : TF[UInt]          = fromDataType(UINT32)
    implicit val uLongEvTF        : TF[ULong]         = fromDataType(UINT64)
    implicit val qByteEvTF        : TF[QByte]         = fromDataType(QINT8)
    implicit val qShortEvTF       : TF[QShort]        = fromDataType(QINT16)
    implicit val qIntEvTF         : TF[QInt]          = fromDataType(QINT32)
    implicit val qUByteEvTF       : TF[QUByte]        = fromDataType(QUINT8)
    implicit val qUShortEvTF      : TF[QUShort]       = fromDataType(QUINT16)
    implicit val resourceEvTF     : TF[Resource]      = fromDataType(RESOURCE)
    implicit val variantEvTF      : TF[Variant]       = fromDataType(VARIANT)
  }

  //  type Float32OrFloat64 = Float | Double
  //  type Float16OrFloat32OrFloat64 = Half | Float | Double
  //  type BFloat16OrFloat32OrFloat64 = TruncatedHalf | Float | Double
  //  type BFloat16OrFloat16OrFloat32 = TruncatedHalf | Half | Float
  //  type Decimal = TruncatedHalf | Half | Float | Double
  //  type Int32OrInt64 = Int | Long
//    type SignedInteger = Byte | Short | Int | Long | UByte | UShort | UInt | ULong
//    type UnsignedInteger = UByte | UShort | UInt | ULong
//    type Integer = SignedInteger | UnsignedInteger
  //  type Real = TruncatedHalf | Half | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong
  //  type Complex = ComplexFloat | ComplexDouble
//    type NotQuantized = TruncatedHalf | Half | Float | Double | Byte | Short | Int | Long | UByte | UShort | UInt | ULong | ComplexFloat | ComplexDouble
//    type Quantized = QByte | QShort | QInt | QUByte | QUShort

  //  type IsFloat32OrFloat64[T] = Union.IsSubtype[T, Float32OrFloat64]
  //  type IsFloat16OrFloat32OrFloat64[T] = Union.IsSubtype[T, Float16OrFloat32OrFloat64]
  //  type IsBFloat16OrFloat32OrFloat64[T] = Union.IsSubtype[T, BFloat16OrFloat32OrFloat64]
  //  type IsBFloat16OrFloat16OrFloat32[T] = Union.IsSubtype[T, BFloat16OrFloat16OrFloat32]
  //  type IsDecimal[T] = Union.IsSubtype[T, Decimal]
  //  type IsInt32OrInt64[T] = Union.IsSubtype[T, Int32OrInt64]
  //  type IsInt32OrInt64OrFloat32OrFloat64[T] = Union.IsSubtype[T, Int32OrInt64 | Float32OrFloat64]
  //  type IsInt32OrInt64OrFloat16OrFloat32OrFloat64[T] = Union.IsSubtype[T, Int32OrInt64 | Float16OrFloat32OrFloat64]
  //  type IsInt32OrInt64OrUInt8[T] = Union.IsSubtype[T, Int32OrInt64 | UByte]
  //  type IsIntOrUInt[T] = Union.IsSubtype[T, IntOrUInt]
  //  type IsStringOrIntOrUInt[T] = Union.IsSubtype[T, String | IntOrUInt]
  //  type IsReal[T] = Union.IsSubtype[T, Real]
  //  type IsComplex[T] = Union.IsSubtype[T, Complex]
  //  type IsNotQuantized[T] = Union.IsSubtype[T, NotQuantized]
  //  type IsQuantized[T] = Union.IsSubtype[T, Quantized]
  //  type IsNumeric[T] = Union.IsSubtype[T, Numeric]
  //  type IsBooleanOrNumeric[T] = Union.IsSubtype[T, BooleanOrNumeric]

  trait IsFloat32OrFloat64[T]

  object IsFloat32OrFloat64 {
    def apply[T: IsFloat32OrFloat64]: IsFloat32OrFloat64[T] = implicitly[IsFloat32OrFloat64[T]]

    implicit val floatEvidence : IsFloat32OrFloat64[Float]  = new IsFloat32OrFloat64[Float] {}
    implicit val doubleEvidence: IsFloat32OrFloat64[Double] = new IsFloat32OrFloat64[Double] {}
  }

  trait IsFloat16OrFloat32OrFloat64[T]

  object IsFloat16OrFloat32OrFloat64 {
    implicit val halfEvidence  : IsFloat16OrFloat32OrFloat64[Half]   = new IsFloat16OrFloat32OrFloat64[Half] {}
    implicit val floatEvidence : IsFloat16OrFloat32OrFloat64[Float]  = new IsFloat16OrFloat32OrFloat64[Float] {}
    implicit val doubleEvidence: IsFloat16OrFloat32OrFloat64[Double] = new IsFloat16OrFloat32OrFloat64[Double] {}

    implicit def float32OrFloat64Evidence[T: IsFloat32OrFloat64]: IsFloat16OrFloat32OrFloat64[T] = new IsFloat16OrFloat32OrFloat64[T] {}
  }

  trait IsBFloat16OrFloat32OrFloat64[T]

  object IsBFloat16OrFloat32OrFloat64 {
    implicit val truncatedHalfEvidence: IsBFloat16OrFloat32OrFloat64[TruncatedHalf] = new IsBFloat16OrFloat32OrFloat64[TruncatedHalf] {}
    implicit val floatEvidence        : IsBFloat16OrFloat32OrFloat64[Float]         = new IsBFloat16OrFloat32OrFloat64[Float] {}
    implicit val doubleEvidence       : IsBFloat16OrFloat32OrFloat64[Double]        = new IsBFloat16OrFloat32OrFloat64[Double] {}

    implicit def float32OrFloat64Evidence[T: IsFloat32OrFloat64]: IsBFloat16OrFloat32OrFloat64[T] = new IsBFloat16OrFloat32OrFloat64[T] {}
  }

  trait IsBFloat16OrFloat16OrFloat32[T]

  object IsBFloat16OrFloat16OrFloat32 {
    def apply[T: IsBFloat16OrFloat16OrFloat32]: IsBFloat16OrFloat16OrFloat32[T] = implicitly[IsBFloat16OrFloat16OrFloat32[T]]

    implicit val truncatedHalfEvidence: IsBFloat16OrFloat16OrFloat32[TruncatedHalf] = new IsBFloat16OrFloat16OrFloat32[TruncatedHalf] {}
    implicit val halfEvidence         : IsBFloat16OrFloat16OrFloat32[Half]          = new IsBFloat16OrFloat16OrFloat32[Half] {}
    implicit val floatEvidence        : IsBFloat16OrFloat16OrFloat32[Float]         = new IsBFloat16OrFloat16OrFloat32[Float] {}
  }

  trait IsDecimal[T]

  object IsDecimal extends IsDecimalPriority0 {
    def apply[T: IsDecimal]: IsDecimal[T] = implicitly[IsDecimal[T]]

    implicit val halfEvidence         : IsDecimal[Half]          = new IsDecimal[Half] {}
    implicit val floatEvidence        : IsDecimal[Float]         = new IsDecimal[Float] {}
    implicit val doubleEvidence       : IsDecimal[Double]        = new IsDecimal[Double] {}
    implicit val truncatedHalfEvidence: IsDecimal[TruncatedHalf] = new IsDecimal[TruncatedHalf] {}

    implicit def float32OrFloat64Evidence[T: IsFloat32OrFloat64]: IsDecimal[T] = new IsDecimal[T] {}
  }

  trait IsDecimalPriority0 {
    implicit def float16OrFloat32OrFloat64Evidence[T: IsFloat16OrFloat32OrFloat64]: IsDecimal[T] = new IsDecimal[T] {}
    implicit def bFloat16OrFloat32OrFloat64Evidence[T: IsBFloat16OrFloat32OrFloat64]: IsDecimal[T] = new IsDecimal[T] {}
    implicit def bFloat16OrFloat16OrFloat32Evidence[T: IsBFloat16OrFloat16OrFloat32]: IsDecimal[T] = new IsDecimal[T] {}
  }

  trait IsInt32OrInt64[T]

  object IsInt32OrInt64 extends IsInt32OrInt64LowPriority {
    def apply[T: IsInt32OrInt64]: IsInt32OrInt64[T] = implicitly[IsInt32OrInt64[T]]

    implicit val intEvidence: IsInt32OrInt64[Int] = new IsInt32OrInt64[Int] {}
  }

  trait IsInt32OrInt64LowPriority {
    implicit val longEvidence: IsInt32OrInt64[Long] = new IsInt32OrInt64[Long] {}
  }

  trait IsInt32OrInt64OrFloat32OrFloat64[T]

  object IsInt32OrInt64OrFloat32OrFloat64 {
    implicit val floatEvidence : IsInt32OrInt64OrFloat32OrFloat64[Float]  = new IsInt32OrInt64OrFloat32OrFloat64[Float] {}
    implicit val doubleEvidence: IsInt32OrInt64OrFloat32OrFloat64[Double] = new IsInt32OrInt64OrFloat32OrFloat64[Double] {}
    implicit val intEvidence   : IsInt32OrInt64OrFloat32OrFloat64[Int]    = new IsInt32OrInt64OrFloat32OrFloat64[Int] {}
    implicit val longEvidence  : IsInt32OrInt64OrFloat32OrFloat64[Long]   = new IsInt32OrInt64OrFloat32OrFloat64[Long] {}

    implicit def float32OrFloat64Evidence[T: IsFloat32OrFloat64]: IsInt32OrInt64OrFloat32OrFloat64[T] = new IsInt32OrInt64OrFloat32OrFloat64[T] {}
    implicit def float16OrFloat32OrFloat64Evidence[T: IsFloat16OrFloat32OrFloat64]: IsInt32OrInt64OrFloat32OrFloat64[T] = new IsInt32OrInt64OrFloat32OrFloat64[T] {}
  }

  trait IsInt32OrInt64OrFloat16OrFloat32OrFloat64[T]

  object IsInt32OrInt64OrFloat16OrFloat32OrFloat64 {
    implicit val halfEvidence  : IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Half]   = new IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Half] {}
    implicit val floatEvidence : IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Float]  = new IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Float] {}
    implicit val doubleEvidence: IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Double] = new IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Double] {}
    implicit val intEvidence   : IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Int]    = new IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Int] {}
    implicit val longEvidence  : IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Long]   = new IsInt32OrInt64OrFloat16OrFloat32OrFloat64[Long] {}

    implicit def float32OrFloat64Evidence[T: IsFloat32OrFloat64]: IsInt32OrInt64OrFloat16OrFloat32OrFloat64[T] = new IsInt32OrInt64OrFloat16OrFloat32OrFloat64[T] {}
    implicit def float16OrFloat32OrFloat64Evidence[T: IsFloat16OrFloat32OrFloat64]: IsInt32OrInt64OrFloat16OrFloat32OrFloat64[T] = new IsInt32OrInt64OrFloat16OrFloat32OrFloat64[T] {}
  }

  trait IsInt32OrInt64OrUInt8[T]

  object IsInt32OrInt64OrUInt8 {
    implicit val uByteEvidence: IsInt32OrInt64OrUInt8[UByte] = new IsInt32OrInt64OrUInt8[UByte] {}
    implicit val intEvidence  : IsInt32OrInt64OrUInt8[Int]   = new IsInt32OrInt64OrUInt8[Int] {}
    implicit val longEvidence : IsInt32OrInt64OrUInt8[Long]  = new IsInt32OrInt64OrUInt8[Long] {}
  }

  trait IsIntOrUInt[T]

  object IsIntOrUInt {
    implicit val byteEvidence  : IsIntOrUInt[Byte]   = new IsIntOrUInt[Byte] {}
    implicit val shortEvidence : IsIntOrUInt[Short]  = new IsIntOrUInt[Short] {}
    implicit val intEvidence   : IsIntOrUInt[Int]    = new IsIntOrUInt[Int] {}
    implicit val longEvidence  : IsIntOrUInt[Long]   = new IsIntOrUInt[Long] {}
    implicit val uByteEvidence : IsIntOrUInt[UByte]  = new IsIntOrUInt[UByte] {}
    implicit val uShortEvidence: IsIntOrUInt[UShort] = new IsIntOrUInt[UShort] {}
    implicit val uIntEvidence  : IsIntOrUInt[UInt]   = new IsIntOrUInt[UInt] {}
    implicit val uLongEvidence : IsIntOrUInt[ULong]  = new IsIntOrUInt[ULong] {}
  }

  trait IsStringOrIntOrUInt[T]

  object IsStringOrIntOrUInt {
    implicit val stringEvidence: IsStringOrIntOrUInt[String] = new IsStringOrIntOrUInt[String] {}
    implicit val byteEvidence  : IsStringOrIntOrUInt[Byte]   = new IsStringOrIntOrUInt[Byte] {}
    implicit val shortEvidence : IsStringOrIntOrUInt[Short]  = new IsStringOrIntOrUInt[Short] {}
    implicit val intEvidence   : IsStringOrIntOrUInt[Int]    = new IsStringOrIntOrUInt[Int] {}
    implicit val longEvidence  : IsStringOrIntOrUInt[Long]   = new IsStringOrIntOrUInt[Long] {}
    implicit val uByteEvidence : IsStringOrIntOrUInt[UByte]  = new IsStringOrIntOrUInt[UByte] {}
    implicit val uShortEvidence: IsStringOrIntOrUInt[UShort] = new IsStringOrIntOrUInt[UShort] {}
    implicit val uIntEvidence  : IsStringOrIntOrUInt[UInt]   = new IsStringOrIntOrUInt[UInt] {}
    implicit val uLongEvidence : IsStringOrIntOrUInt[ULong]  = new IsStringOrIntOrUInt[ULong] {}
  }

  trait IsReal[T]

  object IsReal {
    def apply[T: IsReal]: IsReal[T] = implicitly[IsReal[T]]

    implicit def decimalEvidence[T: IsDecimal]: IsReal[T] = new IsReal[T] {}
    implicit def intOrUIntEvidence[T: IsIntOrUInt]: IsReal[T] = new IsReal[T] {}
  }

  trait IsComplex[T]

  object IsComplex {
    implicit val complexFloatEvidence : IsComplex[ComplexFloat]  = new IsComplex[ComplexFloat] {}
    implicit val complexDoubleEvidence: IsComplex[ComplexDouble] = new IsComplex[ComplexDouble] {}
  }

  trait IsNotQuantized[T]

  object IsNotQuantized extends IsNotQuantizedPriority3 {
    def apply[T: IsNotQuantized]: IsNotQuantized[T] = implicitly[IsNotQuantized[T]]

    implicit val halfEvidence         : IsNotQuantized[Half]          = new IsNotQuantized[Half] {}
    implicit val floatEvidence        : IsNotQuantized[Float]         = new IsNotQuantized[Float] {}
    implicit val doubleEvidence       : IsNotQuantized[Double]        = new IsNotQuantized[Double] {}
    implicit val complexFloatEvidence : IsNotQuantized[ComplexFloat]  = new IsNotQuantized[ComplexFloat] {}
    implicit val complexDoubleEvidence: IsNotQuantized[ComplexDouble] = new IsNotQuantized[ComplexDouble] {}
    implicit val byteEvidence         : IsNotQuantized[Byte]          = new IsNotQuantized[Byte] {}
    implicit val shortEvidence        : IsNotQuantized[Short]         = new IsNotQuantized[Short] {}
    implicit val intEvidence          : IsNotQuantized[Int]           = new IsNotQuantized[Int] {}
    implicit val longEvidence         : IsNotQuantized[Long]          = new IsNotQuantized[Long] {}
    implicit val uByteEvidence        : IsNotQuantized[UByte]         = new IsNotQuantized[UByte] {}
    implicit val uShortEvidence       : IsNotQuantized[UShort]        = new IsNotQuantized[UShort] {}
    implicit val uIntEvidence         : IsNotQuantized[UInt]          = new IsNotQuantized[UInt] {}
    implicit val uLongEvidence        : IsNotQuantized[ULong]         = new IsNotQuantized[ULong] {}
  }

  trait IsNotQuantizedPriority3 extends IsNotQuantizedPriority2 {
    implicit def float32OrFloat64Evidence[T: IsFloat32OrFloat64]: IsNotQuantized[T] = new IsNotQuantized[T] {}
    implicit def int32OrInt64Evidence[T: IsInt32OrInt64]: IsNotQuantized[T] = new IsNotQuantized[T] {}
  }

  trait IsNotQuantizedPriority2 extends IsNotQuantizedPriority1 {
    implicit def float16OrFloat32OrFloat64Evidence[T: IsFloat16OrFloat32OrFloat64]: IsNotQuantized[T] = new IsNotQuantized[T] {}
  }

  trait IsNotQuantizedPriority1 extends IsNotQuantizedPriority0 {
    implicit def int32OrInt64OrFloat16OrFloat32OrFloat64Evidence[T: IsInt32OrInt64OrFloat16OrFloat32OrFloat64]: IsNotQuantized[T] = new IsNotQuantized[T] {}
  }

  trait IsNotQuantizedPriority0 {
    implicit def realEvidence[T: IsReal]: IsNotQuantized[T] = new IsNotQuantized[T] {}
    implicit def complexEvidence[T: IsComplex]: IsNotQuantized[T] = new IsNotQuantized[T] {}
  }

  trait IsQuantized[T]

  object IsQuantized {
    implicit val qByteEvidence  : IsQuantized[QByte]   = new IsQuantized[QByte] {}
    implicit val qShortEvidence : IsQuantized[QShort]  = new IsQuantized[QShort] {}
    implicit val qIntEvidence   : IsQuantized[QInt]    = new IsQuantized[QInt] {}
    implicit val qUByteEvidence : IsQuantized[QUByte]  = new IsQuantized[QUByte] {}
    implicit val qUShortEvidence: IsQuantized[QUShort] = new IsQuantized[QUShort] {}
  }

  trait IsNumeric[T]

  object IsNumeric extends IsNumericPriority0 {
    def apply[T: IsNumeric]: IsNumeric[T] = implicitly[IsNumeric[T]]

    implicit val halfEvidence         : IsNumeric[Half]          = new IsNumeric[Half] {}
    implicit val floatEvidence        : IsNumeric[Float]         = new IsNumeric[Float] {}
    implicit val doubleEvidence       : IsNumeric[Double]        = new IsNumeric[Double] {}
    implicit val truncatedHalfEvidence: IsNumeric[TruncatedHalf] = new IsNumeric[TruncatedHalf] {}
    implicit val complexFloatEvidence : IsNumeric[ComplexFloat]  = new IsNumeric[ComplexFloat] {}
    implicit val complexDoubleEvidence: IsNumeric[ComplexDouble] = new IsNumeric[ComplexDouble] {}
    implicit val byteEvidence         : IsNumeric[Byte]          = new IsNumeric[Byte] {}
    implicit val shortEvidence        : IsNumeric[Short]         = new IsNumeric[Short] {}
    implicit val intEvidence          : IsNumeric[Int]           = new IsNumeric[Int] {}
    implicit val longEvidence         : IsNumeric[Long]          = new IsNumeric[Long] {}
    implicit val uByteEvidence        : IsNumeric[UByte]         = new IsNumeric[UByte] {}
    implicit val uShortEvidence       : IsNumeric[UShort]        = new IsNumeric[UShort] {}
    implicit val uIntEvidence         : IsNumeric[UInt]          = new IsNumeric[UInt] {}
    implicit val uLongEvidence        : IsNumeric[ULong]         = new IsNumeric[ULong] {}
    implicit val qByteEvidence        : IsNumeric[QByte]         = new IsNumeric[QByte] {}
    implicit val qShortEvidence       : IsNumeric[QShort]        = new IsNumeric[QShort] {}
    implicit val qIntEvidence         : IsNumeric[QInt]          = new IsNumeric[QInt] {}
    implicit val qUByteEvidence       : IsNumeric[QUByte]        = new IsNumeric[QUByte] {}
    implicit val qUShortEvidence      : IsNumeric[QUShort]       = new IsNumeric[QUShort] {}

    implicit def int32OrInt64Evidence[T: IsInt32OrInt64]: IsNumeric[T] = new IsNumeric[T] {}
  }

  trait IsNumericPriority0 {
    implicit def notQuantizedEvidence[T: IsNotQuantized]: IsNumeric[T] = new IsNumeric[T] {}
    implicit def quantizedEvidence[T: IsQuantized]: IsNumeric[T] = new IsNumeric[T] {}
  }

  trait IsBooleanOrNumeric[T]

  object IsBooleanOrNumeric {
    implicit val booleanEvidence      : IsBooleanOrNumeric[Boolean]       = new IsBooleanOrNumeric[Boolean] {}
    implicit val halfEvidence         : IsBooleanOrNumeric[Half]          = new IsBooleanOrNumeric[Half] {}
    implicit val floatEvidence        : IsBooleanOrNumeric[Float]         = new IsBooleanOrNumeric[Float] {}
    implicit val doubleEvidence       : IsBooleanOrNumeric[Double]        = new IsBooleanOrNumeric[Double] {}
    implicit val truncatedHalfEvidence: IsBooleanOrNumeric[TruncatedHalf] = new IsBooleanOrNumeric[TruncatedHalf] {}
    implicit val complexFloatEvidence : IsBooleanOrNumeric[ComplexFloat]  = new IsBooleanOrNumeric[ComplexFloat] {}
    implicit val complexDoubleEvidence: IsBooleanOrNumeric[ComplexDouble] = new IsBooleanOrNumeric[ComplexDouble] {}
    implicit val byteEvidence         : IsBooleanOrNumeric[Byte]          = new IsBooleanOrNumeric[Byte] {}
    implicit val shortEvidence        : IsBooleanOrNumeric[Short]         = new IsBooleanOrNumeric[Short] {}
    implicit val intEvidence          : IsBooleanOrNumeric[Int]           = new IsBooleanOrNumeric[Int] {}
    implicit val longEvidence         : IsBooleanOrNumeric[Long]          = new IsBooleanOrNumeric[Long] {}
    implicit val uByteEvidence        : IsBooleanOrNumeric[UByte]         = new IsBooleanOrNumeric[UByte] {}
    implicit val uShortEvidence       : IsBooleanOrNumeric[UShort]        = new IsBooleanOrNumeric[UShort] {}
    implicit val uIntEvidence         : IsBooleanOrNumeric[UInt]          = new IsBooleanOrNumeric[UInt] {}
    implicit val uLongEvidence        : IsBooleanOrNumeric[ULong]         = new IsBooleanOrNumeric[ULong] {}
    implicit val qByteEvidence        : IsBooleanOrNumeric[QByte]         = new IsBooleanOrNumeric[QByte] {}
    implicit val qShortEvidence       : IsBooleanOrNumeric[QShort]        = new IsBooleanOrNumeric[QShort] {}
    implicit val qIntEvidence         : IsBooleanOrNumeric[QInt]          = new IsBooleanOrNumeric[QInt] {}
    implicit val qUByteEvidence       : IsBooleanOrNumeric[QUByte]        = new IsBooleanOrNumeric[QUByte] {}
    implicit val qUShortEvidence      : IsBooleanOrNumeric[QUShort]       = new IsBooleanOrNumeric[QUShort] {}

    implicit def int32OrInt64Evidence[T: IsInt32OrInt64]: IsNumeric[T] = new IsNumeric[T] {}
    implicit def notQuantizedEvidence[T: IsNotQuantized]: IsBooleanOrNumeric[T] = new IsBooleanOrNumeric[T] {}
    implicit def quantizedEvidence[T: IsQuantized]: IsBooleanOrNumeric[T] = new IsBooleanOrNumeric[T] {}
    implicit def numericEvidence[T: IsNumeric]: IsBooleanOrNumeric[T] = new IsBooleanOrNumeric[T] {}
  }

  //endregion Type Traits
}
