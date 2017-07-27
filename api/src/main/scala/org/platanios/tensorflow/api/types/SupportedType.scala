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

package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException

import spire.algebra._
import spire.math._
import spire.std._

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait SupportedType[@specialized T]
    extends Any {
  @inline def dataType: DataType

  @throws[UnsupportedOperationException]
  @inline def cast[R: SupportedType](value: R, dataType: DataType): T = {
    throw InvalidDataTypeException("The Scala type of this data type is not supported.")
  }
}

sealed trait FixedSizeSupportedType[@specialized T]
    extends Any
        with SupportedType[T] {
  @inline override def dataType: FixedSizeDataType
}

sealed trait NumericSupportedType[@specialized T]
    extends Any
        with FixedSizeSupportedType[T]
        with Rig[T]
        with Order[T]
        // TODO: Add Trig[T] here.
        with ConvertableFrom[T]
        with ConvertableTo[T] {
  @inline override def dataType: NumericDataType
}

sealed trait SignedNumericSupportedType[@specialized T]
    extends Any
        with NumericSupportedType[T]
        with Signed[T] {
  @inline override def dataType: SignedNumericDataType
}

// TODO: [TYPES] Separate integral types from the rest.

sealed trait RealNumericSupportedType[@specialized T]
    extends Any
        with SignedNumericSupportedType[T]
        with Ring[T]
        with IsRational[T] {
  @inline override def dataType: RealNumericDataType
}

sealed trait ComplexNumericSupportedType[@specialized T]
    extends Any
        with SignedNumericSupportedType[T]
        with Trig[T] {
  @inline override def dataType: ComplexNumericDataType
}

object SupportedType {
  class SupportedTypeOps[T](val value: T) extends AnyVal {
    @inline def dataType(implicit evidence: SupportedType[T]): DataType = evidence.dataType

    @inline def cast(dataType: DataType)
        (implicit evidence: SupportedType[T],
            dataTypeEvidence: SupportedType[dataType.ScalaType]): dataType.ScalaType = {
      dataTypeEvidence.cast(value, dataType)
    }
  }

  @inline final def apply[T](implicit evidence: SupportedType[T]): SupportedType[T] = evidence

  private[api] trait Implicits {
    implicit def toSupportedTypeOps[@specialized T](value: T): SupportedTypeOps[T] = {
      new SupportedTypeOps(value)
    }

    implicit final val StringIsSupportedType : SupportedType[String]            = new StringIsSupportedType
    implicit final val BooleanIsSupportedType: FixedSizeSupportedType[Boolean]  = new BooleanIsSupportedType
    implicit final val FloatIsSupportedType  : RealNumericSupportedType[Float]  = new FloatIsSupportedType
    implicit final val DoubleIsSupportedType : RealNumericSupportedType[Double] = new DoubleIsSupportedType
    implicit final val ByteIsSupportedType   : RealNumericSupportedType[Byte]   = new ByteIsSupportedType
    implicit final val ShortIsSupportedType  : RealNumericSupportedType[Short]  = new ShortIsSupportedType
    implicit final val IntIsSupportedType    : RealNumericSupportedType[Int]    = new IntIsSupportedType
    implicit final val LongIsSupportedType   : RealNumericSupportedType[Long]   = new LongIsSupportedType
    implicit final val UByteIsSupportedType  : NumericSupportedType[UByte]      = new UByteIsSupportedType
    implicit final val UShortIsSupportedType : NumericSupportedType[UShort]     = new UShortIsSupportedType
  }

  private[api] object Implicits extends Implicits
}

private[types] class StringIsSupportedType extends SupportedType[String] {
  @inline override def dataType: DataType = STRING
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): String = value.toString
}

private[types] class BooleanIsSupportedType extends FixedSizeSupportedType[Boolean] {
  @inline override def dataType: FixedSizeDataType = BOOLEAN
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): Boolean = value match {
    case value: Boolean => value
    case _ => throw InvalidDataTypeException("Cannot convert the provided value to a boolean.")
  }
}

private[types] class FloatIsSupportedType
    extends RealNumericSupportedType[Float]
        with FloatIsField
        with FloatIsReal
        with ConvertableFromFloat
        with ConvertableToFloat {
  @inline override def dataType: RealNumericDataType = FLOAT32
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): Float = value match {
    case value: Boolean => if (value) 1.0f else 0.0f
    case value: Float => value.toFloat
    case value: Double => value.toFloat
    case value: Byte => value.toFloat
    case value: Short => value.toFloat
    case value: Int => value.toFloat
    case value: Long => value.toFloat
    case value: UByte => value.toFloat
    case value: UShort => value.toFloat
    case _ => throw InvalidDataTypeException("Cannot convert the provided value to a float.")
  }
}

private[types] class DoubleIsSupportedType
    extends RealNumericSupportedType[Double]
        with DoubleIsField
        with DoubleIsReal
        with ConvertableFromDouble
        with ConvertableToDouble {
  @inline override def dataType: RealNumericDataType = FLOAT64
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): Double = value match {
    case value: Boolean => if (value) 1.0 else 0.0
    case value: Float => value.toDouble
    case value: Double => value.toDouble
    case value: Byte => value.toDouble
    case value: Short => value.toDouble
    case value: Int => value.toDouble
    case value: Long => value.toDouble
    case value: UByte => value.toDouble
    case value: UShort => value.toDouble
    case _ => throw InvalidDataTypeException("Cannot convert the provided value to a double.")
  }
}

private[types] class ByteIsSupportedType
    extends RealNumericSupportedType[Byte]
        with ByteIsEuclideanRing
        with ByteIsReal
        with ConvertableFromByte
        with ConvertableToByte {
  @inline override def dataType: RealNumericDataType = INT8
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): Byte = value match {
    case value: Boolean => if (value) 1 else 0
    case value: Float => value.toByte
    case value: Double => value.toByte
    case value: Byte => value.toByte
    case value: Short => value.toByte
    case value: Int => value.toByte
    case value: Long => value.toByte
    case value: UByte => value.toByte
    case value: UShort => value.toByte
    case _ => throw InvalidDataTypeException("Cannot convert the provided value to a byte.")
  }
}

private[types] class ShortIsSupportedType
    extends RealNumericSupportedType[Short]
        with ShortIsEuclideanRing
        with ShortIsReal
        with ConvertableFromShort
        with ConvertableToShort {
  @inline override def dataType: RealNumericDataType = INT16
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): Short = value match {
    case value: Boolean => if (value) 1 else 0
    case value: Float => value.toShort
    case value: Double => value.toShort
    case value: Byte => value.toShort
    case value: Short => value.toShort
    case value: Int => value.toShort
    case value: Long => value.toShort
    case value: UByte => value.toShort
    case value: UShort => value.toShort
    case _ => throw InvalidDataTypeException("Cannot convert the provided value to a short.")
  }
}

private[types] class IntIsSupportedType
    extends RealNumericSupportedType[Int]
        with IntIsEuclideanRing
        with IntIsReal
        with ConvertableFromInt
        with ConvertableToInt {
  @inline override def dataType: RealNumericDataType = INT32
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): Int = value match {
    case value: Boolean => if (value) 1 else 0
    case value: Float => value.toInt
    case value: Double => value.toInt
    case value: Byte => value.toInt
    case value: Short => value.toInt
    case value: Int => value.toInt
    case value: Long => value.toInt
    case value: UByte => value.toInt
    case value: UShort => value.toInt
    case _ => throw InvalidDataTypeException("Cannot convert the provided value to an integer.")
  }
}

private[types] class LongIsSupportedType
    extends RealNumericSupportedType[Long]
        with LongIsEuclideanRing
        with LongIsReal
        with ConvertableFromLong
        with ConvertableToLong {
  @inline override def dataType: RealNumericDataType = INT64
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): Long = value match {
    case value: Boolean => if (value) 1L else 0L
    case value: Float => value.toLong
    case value: Double => value.toLong
    case value: Byte => value.toLong
    case value: Short => value.toLong
    case value: Int => value.toLong
    case value: Long => value.toLong
    case value: UByte => value.toLong
    case value: UShort => value.toLong
    case _ => throw InvalidDataTypeException("Cannot convert the provided value to a long.")
  }
}

private[types] class UByteIsSupportedType
    extends NumericSupportedType[UByte]
        with UByteIsCRig
        with UByteIsReal
        with ConvertableFromUByte
        with ConvertableToUByte {
  @inline override def dataType: NumericDataType = UINT8
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): UByte = value match {
    case value: Boolean => if (value) UByte(1) else UByte(0)
    case value: Float => UByte(value.toInt)
    case value: Double => UByte(value.toInt)
    case value: Byte => UByte(value)
    case value: Short => UByte(value)
    case value: Int => UByte(value)
    case value: Long => UByte(value.toInt)
    case value: UByte => value
    case value: UShort => UByte(value.toInt)
    case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned byte.")
  }
}

private[this] trait UByteIsCRig extends CRig[UByte] {
  override def zero: UByte = UByte(0)
  override def one: UByte = UByte(1)
  override def plus(a: UByte, b: UByte): UByte = a + b
  override def times(a: UByte, b: UByte): UByte = a * b
  override def pow(a: UByte, b: Int): UByte = {
    if (b < 0)
      throw new IllegalArgumentException("negative exponent: %s" format b)
    a ** UByte(b)
  }
}

private[this] trait UByteSigned extends SignedAdditiveCMonoid[UByte] {
  override def eqv(x: UByte, y: UByte): Boolean = x == y
  override def neqv(x: UByte, y: UByte): Boolean = x != y
  override def gt(x: UByte, y: UByte): Boolean = x > y
  override def gteqv(x: UByte, y: UByte): Boolean = x >= y
  override def lt(x: UByte, y: UByte): Boolean = x < y
  override def lteqv(x: UByte, y: UByte): Boolean = x <= y
  override def compare(x: UByte, y: UByte): Int = if (x < y) -1 else if (x > y) 1 else 0
  override def abs(x: UByte): UByte = x
}

private[this] trait UByteIsReal extends IsIntegral[UByte] with UByteSigned {
  def toDouble(n: UByte): Double = n.toDouble
  def toBigInt(n: UByte): BigInt = n.toBigInt
}

private[types] class UShortIsSupportedType
    extends NumericSupportedType[UShort]
        with UShortIsCRig
        with UShortIsReal
        with ConvertableFromUShort
        with ConvertableToUShort {
  @inline override def dataType: NumericDataType = UINT16
  @inline override def cast[R: SupportedType](value: R, dataType: DataType): UShort = value match {
    case value: Boolean => if (value) UShort(1) else UShort(0)
    case value: Float => UShort(value.toInt)
    case value: Double => UShort(value.toInt)
    case value: Byte => UShort(value)
    case value: Short => UShort(value)
    case value: Int => UShort(value)
    case value: Long => UShort(value.toInt)
    case value: UByte => UShort(value.toShort)
    case value: UShort => value
    case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned short.")
  }
}

private[this] trait UShortIsCRig extends CRig[UShort] {
  override def zero: UShort = UShort(0)
  override def one: UShort = UShort(1)
  override def plus(a: UShort, b: UShort): UShort = a + b
  override def times(a: UShort, b: UShort): UShort = a * b
  override def pow(a: UShort, b: Int): UShort = {
    if (b < 0)
      throw new IllegalArgumentException("negative exponent: %s" format b)
    a ** UShort(b)
  }
}

private[this] trait UShortSigned extends SignedAdditiveCMonoid[UShort] {
  override def eqv(x: UShort, y: UShort): Boolean = x == y
  override def neqv(x: UShort, y: UShort): Boolean = x != y
  override def gt(x: UShort, y: UShort): Boolean = x > y
  override def gteqv(x: UShort, y: UShort): Boolean = x >= y
  override def lt(x: UShort, y: UShort): Boolean = x < y
  override def lteqv(x: UShort, y: UShort): Boolean = x <= y
  override def compare(x: UShort, y: UShort): Int = if (x < y) -1 else if (x > y) 1 else 0
  override def abs(x: UShort): UShort = x
}

private[this] trait UShortIsReal extends IsIntegral[UShort] with UShortSigned {
  def toDouble(n: UShort): Double = n.toDouble
  def toBigInt(n: UShort): BigInt = n.toBigInt
}
