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

  type STRING = types.DataType[String]
  type BOOLEAN = types.DataType[Boolean]
  type FLOAT16 = types.DataType[Half]
  type FLOAT32 = types.DataType[Float]
  type FLOAT64 = types.DataType[Double]
  type BFLOAT16 = types.DataType[TruncatedHalf]
  type COMPLEX64 = types.DataType[ComplexFloat]
  type COMPLEX128 = types.DataType[ComplexDouble]
  type INT8 = types.DataType[Byte]
  type INT16 = types.DataType[Short]
  type INT32 = types.DataType[Int]
  type INT64 = types.DataType[Long]
  type UINT8 = types.DataType[UByte]
  type UINT16 = types.DataType[UShort]
  type UINT32 = types.DataType[UInt]
  type UINT64 = types.DataType[ULong]
  type QINT8 = types.DataType[QByte]
  type QINT16 = types.DataType[QShort]
  type QINT32 = types.DataType[QInt]
  type QUINT8 = types.DataType[QUByte]
  type QUINT16 = types.DataType[QUShort]
  type RESOURCE = types.DataType[Long]
  type VARIANT = types.DataType[Long]

  val STRING    : STRING     = types.DataType.STRING
  val BOOLEAN   : BOOLEAN    = types.DataType.BOOLEAN
  val FLOAT16   : FLOAT16    = types.DataType.FLOAT16
  val FLOAT32   : FLOAT32    = types.DataType.FLOAT32
  val FLOAT64   : FLOAT64    = types.DataType.FLOAT64
  val BFLOAT16  : BFLOAT16   = types.DataType.BFLOAT16
  val COMPLEX64 : COMPLEX64  = types.DataType.COMPLEX64
  val COMPLEX128: COMPLEX128 = types.DataType.COMPLEX128
  val INT8      : INT8       = types.DataType.INT8
  val INT16     : INT16      = types.DataType.INT16
  val INT32     : INT32      = types.DataType.INT32
  val INT64     : INT64      = types.DataType.INT64
  val UINT8     : UINT8      = types.DataType.UINT8
  val UINT16    : UINT16     = types.DataType.UINT16
  val UINT32    : UINT32     = types.DataType.UINT32
  val UINT64    : UINT64     = types.DataType.UINT64
  val QINT8     : QINT8      = types.DataType.QINT8
  val QINT16    : QINT16     = types.DataType.QINT16
  val QINT32    : QINT32     = types.DataType.QINT32
  val QUINT8    : QUINT8     = types.DataType.QUINT8
  val QUINT16   : QUINT16    = types.DataType.QUINT16
  val RESOURCE  : RESOURCE   = types.DataType.RESOURCE
  val VARIANT   : VARIANT    = types.DataType.VARIANT

  //region Type Traits

  // TODO: Complete/generalize this (potentially using union types).

  trait IsFloat32OrFloat64[T]

  object IsFloat32OrFloat64 {
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
    implicit val truncatedHalfEvidence: IsBFloat16OrFloat16OrFloat32[TruncatedHalf] = new IsBFloat16OrFloat16OrFloat32[TruncatedHalf] {}
    implicit val halfEvidence         : IsBFloat16OrFloat16OrFloat32[Half]          = new IsBFloat16OrFloat16OrFloat32[Half] {}
    implicit val floatEvidence        : IsBFloat16OrFloat16OrFloat32[Float]         = new IsBFloat16OrFloat16OrFloat32[Float] {}
  }

  trait IsDecimal[T]

  object IsDecimal {
    implicit val halfEvidence         : IsDecimal[Half]          = new IsDecimal[Half] {}
    implicit val floatEvidence        : IsDecimal[Float]         = new IsDecimal[Float] {}
    implicit val doubleEvidence       : IsDecimal[Double]        = new IsDecimal[Double] {}
    implicit val truncatedHalfEvidence: IsDecimal[TruncatedHalf] = new IsDecimal[TruncatedHalf] {}

    implicit def float32OrFloat64Evidence[T: IsFloat32OrFloat64]: IsDecimal[T] = new IsDecimal[T] {}
    implicit def float16OrFloat32OrFloat64Evidence[T: IsFloat16OrFloat32OrFloat64]: IsDecimal[T] = new IsDecimal[T] {}
    implicit def bFloat16OrFloat32OrFloat64Evidence[T: IsBFloat16OrFloat32OrFloat64]: IsDecimal[T] = new IsDecimal[T] {}
    implicit def bFloat16OrFloat16OrFloat32Evidence[T: IsBFloat16OrFloat16OrFloat32]: IsDecimal[T] = new IsDecimal[T] {}
  }

  trait IsInt32OrInt64[T]

  object IsInt32OrInt64 extends IsInt32OrInt64LowPriority {
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

  trait IsReal[T]

  object IsReal {
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
    implicit val halfEvidence         : IsNotQuantized[Half]          = new IsNotQuantized[Half] {}
    implicit val floatEvidence        : IsNotQuantized[Float]         = new IsNotQuantized[Float] {}
    implicit val doubleEvidence       : IsNotQuantized[Double]        = new IsNotQuantized[Double] {}
    implicit val truncatedHalfEvidence: IsNotQuantized[TruncatedHalf] = new IsNotQuantized[TruncatedHalf] {}
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

  object IsNumeric {
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
