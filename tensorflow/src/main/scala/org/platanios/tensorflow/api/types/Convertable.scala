// Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package org.platanios.tensorflow.api.types

import java.math.MathContext

import spire.math._

/** Helper trait implementations borrowed from spire.
  * 
  * @author Emmanouil Antonios Platanios
  */
private[types] trait ConvertableToFloat extends ConvertableTo[Float] {
  def fromByte(a: Byte): Float = a.toFloat
  def fromShort(a: Short): Float = a.toFloat
  // def fromInt(a: Int): Float = a.toFloat
  def fromLong(a: Long): Float = a.toFloat
  def fromFloat(a: Float): Float = a
  // def fromDouble(a: Double): Float = a.toFloat
  // def fromBigInt(a: BigInt): Float = a.toFloat
  def fromBigDecimal(a: BigDecimal): Float = a.toFloat
  def fromRational(a: Rational): Float = a.toBigDecimal(MathContext.DECIMAL64).toFloat
  def fromAlgebraic(a: Algebraic): Float = a.toFloat
  def fromReal(a: Real): Float = a.toFloat

  def fromType[B: ConvertableFrom](b: B): Float = ConvertableFrom[B].toFloat(b)
}

private[types] trait ConvertableToDouble extends ConvertableTo[Double] {
  def fromByte(a: Byte): Double = a.toDouble
  def fromShort(a: Short): Double = a.toDouble
  // def fromInt(a: Int): Double = a.toDouble
  def fromLong(a: Long): Double = a.toDouble
  def fromFloat(a: Float): Double = a.toDouble
  // def fromDouble(a: Double): Double = a
  // def fromBigInt(a: BigInt): Double = a.toDouble
  def fromBigDecimal(a: BigDecimal): Double = a.toDouble
  def fromRational(a: Rational): Double = a.toBigDecimal(MathContext.DECIMAL64).toDouble
  def fromAlgebraic(a: Algebraic): Double = a.toDouble
  def fromReal(a: Real): Double = a.toDouble

  def fromType[B: ConvertableFrom](b: B): Double = ConvertableFrom[B].toDouble(b)
}

private[types] trait ConvertableToComplex[A] extends ConvertableTo[Complex[A]] {
  implicit def algebra: Integral[A]

  def fromByte(a: Byte): Complex[A] = Complex(algebra.fromByte(a), algebra.zero)
  def fromShort(a: Short): Complex[A] = Complex(algebra.fromShort(a), algebra.zero)
  def fromInt(a: Int): Complex[A] = Complex(algebra.fromInt(a), algebra.zero)
  def fromLong(a: Long): Complex[A] = Complex(algebra.fromLong(a), algebra.zero)
  def fromFloat(a: Float): Complex[A] = Complex(algebra.fromFloat(a), algebra.zero)
  def fromDouble(a: Double): Complex[A] = Complex(algebra.fromDouble(a), algebra.zero)
  def fromBigInt(a: BigInt): Complex[A] = Complex(algebra.fromBigInt(a), algebra.zero)
  def fromBigDecimal(a: BigDecimal): Complex[A] = Complex(algebra.fromBigDecimal(a), algebra.zero)
  def fromRational(a: Rational): Complex[A] = Complex(algebra.fromRational(a), algebra.zero)
  def fromAlgebraic(a: Algebraic): Complex[A] = Complex(algebra.fromAlgebraic(a), algebra.zero)
  def fromReal(a: Real): Complex[A] = Complex(algebra.fromReal(a), algebra.zero)

  def fromType[B: ConvertableFrom](b: B): Complex[A] = Complex(algebra.fromType(b), algebra.zero)
}

private[types] trait ConvertableToByte extends ConvertableTo[Byte] {
  def fromByte(a: Byte): Byte = a
  def fromShort(a: Short): Byte = a.toByte
  // def fromInt(a: Int): Byte = a.toByte
  def fromLong(a: Long): Byte = a.toByte
  def fromFloat(a: Float): Byte = a.toByte
  def fromDouble(a: Double): Byte = a.toByte
  // def fromBigInt(a: BigInt): Byte = a.toByte
  def fromBigDecimal(a: BigDecimal): Byte = a.toByte
  def fromRational(a: Rational): Byte = a.toBigInt.toByte
  def fromAlgebraic(a: Algebraic): Byte = a.toByte
  def fromReal(a: Real): Byte = a.toByte

  def fromType[B: ConvertableFrom](b: B): Byte = ConvertableFrom[B].toByte(b)
}

private[types] trait ConvertableToShort extends ConvertableTo[Short] {
  def fromByte(a: Byte): Short = a.toShort
  def fromShort(a: Short): Short = a
  // def fromInt(a: Int): Short = a.toShort
  def fromLong(a: Long): Short = a.toShort
  def fromFloat(a: Float): Short = a.toShort
  def fromDouble(a: Double): Short = a.toShort
  // def fromBigInt(a: BigInt): Short = a.toShort
  def fromBigDecimal(a: BigDecimal): Short = a.toShort
  def fromRational(a: Rational): Short = a.toBigInt.toShort
  def fromAlgebraic(a: Algebraic): Short = a.toShort
  def fromReal(a: Real): Short = a.toShort

  def fromType[B: ConvertableFrom](b: B): Short = ConvertableFrom[B].toShort(b)
}

private[types] trait ConvertableToInt extends ConvertableTo[Int] {
  def fromByte(a: Byte): Int = a.toInt
  def fromShort(a: Short): Int = a.toInt
  // def fromInt(a: Int): Int = a
  def fromLong(a: Long): Int = a.toInt
  def fromFloat(a: Float): Int = a.toInt
  def fromDouble(a: Double): Int = a.toInt
  // def fromBigInt(a: BigInt): Int = a.toInt
  def fromBigDecimal(a: BigDecimal): Int = a.toInt
  def fromRational(a: Rational): Int = a.toBigInt.toInt
  def fromAlgebraic(a: Algebraic): Int = a.toInt
  def fromReal(a: Real): Int = a.toInt

  def fromType[B: ConvertableFrom](b: B): Int = ConvertableFrom[B].toInt(b)
}

private[types] trait ConvertableToLong extends ConvertableTo[Long] {
  def fromByte(a: Byte): Long = a.toLong
  def fromShort(a: Short): Long = a.toLong
  // def fromInt(a: Int): Long = a.toLong
  def fromLong(a: Long): Long = a
  def fromFloat(a: Float): Long = a.toLong
  def fromDouble(a: Double): Long = a.toLong
  // def fromBigInt(a: BigInt): Long = a.toLong
  def fromBigDecimal(a: BigDecimal): Long = a.toLong
  def fromRational(a: Rational): Long = a.toBigInt.toLong
  def fromAlgebraic(a: Algebraic): Long = a.toLong
  def fromReal(a: Real): Long = a.toLong

  def fromType[B: ConvertableFrom](b: B): Long = ConvertableFrom[B].toLong(b)
}

private[types] trait ConvertableToUByte extends ConvertableTo[UByte] {
  def fromByte(a: Byte): UByte = UByte(a)
  def fromShort(a: Short): UByte = UByte(a.toByte)
  def fromInt(a: Int): UByte = UByte(a.toByte)
  def fromLong(a: Long): UByte = UByte(a.toByte)
  def fromFloat(a: Float): UByte = UByte(a.toByte)
  def fromDouble(a: Double): UByte = UByte(a.toByte)
  def fromBigInt(a: BigInt): UByte = UByte(a.toByte)
  def fromBigDecimal(a: BigDecimal): UByte = UByte(a.toByte)
  def fromRational(a: Rational): UByte = UByte(a.toBigInt.toByte)
  def fromAlgebraic(a: Algebraic): UByte = UByte(a.toByte)
  def fromReal(a: Real): UByte = UByte(a.toByte)

  def fromType[B: ConvertableFrom](b: B): UByte = UByte(ConvertableFrom[B].toByte(b))
}

private[types] trait ConvertableToUShort extends ConvertableTo[UShort] {
  def fromByte(a: Byte): UShort = UShort(a.toShort)
  def fromShort(a: Short): UShort = UShort(a)
  def fromInt(a: Int): UShort = UShort(a.toShort)
  def fromLong(a: Long): UShort = UShort(a.toShort)
  def fromFloat(a: Float): UShort = UShort(a.toShort)
  def fromDouble(a: Double): UShort = UShort(a.toShort)
  def fromBigInt(a: BigInt): UShort = UShort(a.toShort)
  def fromBigDecimal(a: BigDecimal): UShort = UShort(a.toShort)
  def fromRational(a: Rational): UShort = UShort(a.toBigInt.toShort)
  def fromAlgebraic(a: Algebraic): UShort = UShort(a.toShort)
  def fromReal(a: Real): UShort = UShort(a.toShort)

  def fromType[B: ConvertableFrom](b: B): UShort = UShort(ConvertableFrom[B].toShort(b))
}

private[types] trait ConvertableFromFloat extends ConvertableFrom[Float] {
  def toByte(a: Float): Byte = a.toByte
  def toShort(a: Float): Short = a.toShort
  def toInt(a: Float): Int = a.toInt
  def toLong(a: Float): Long = a.toLong
  def toFloat(a: Float): Float = a
  // def toDouble(a: Float): Double = a.toDouble
  def toBigInt(a: Float): BigInt = BigInt(a.toLong)
  def toBigDecimal(a: Float): BigDecimal = BigDecimal(a.toDouble)
  // def toRational(a: Float): Rational = Rational(a)
  // def toAlgebraic(a: Float): Algebraic = Algebraic(a)
  // def toReal(a: Float): Real = Real(a)
  def toNumber(a: Float): Number = Number(a)

  def toType[B: ConvertableTo](a: Float): B = ConvertableTo[B].fromFloat(a)
  def toString(a: Float): String = a.toString
}

private[types] trait ConvertableFromDouble extends ConvertableFrom[Double] {
  def toByte(a: Double): Byte = a.toByte
  def toShort(a: Double): Short = a.toShort
  def toInt(a: Double): Int = a.toInt
  def toLong(a: Double): Long = a.toLong
  def toFloat(a: Double): Float = a.toFloat
  // def toDouble(a: Double): Double = a
  def toBigInt(a: Double): BigInt = BigInt(a.toLong)
  def toBigDecimal(a: Double): BigDecimal = BigDecimal(a)
  // def toRational(a: Double): Rational = Rational(a)
  // def toAlgebraic(a: Double): Algebraic = Algebraic(a)
  // def toReal(a: Double): Real = Real(a)
  def toNumber(a: Double): Number = Number(a)

  def toType[B: ConvertableTo](a: Double): B = ConvertableTo[B].fromDouble(a)
  def toString(a: Double): String = a.toString
}

private[types] trait ConvertableFromComplex[A] extends ConvertableFrom[Complex[A]] {
  def algebra: Integral[A]

  def toByte(a: Complex[A]): Byte = algebra.toByte(a.real)
  def toShort(a: Complex[A]): Short = algebra.toShort(a.real)
  def toInt(a: Complex[A]): Int = algebra.toInt(a.real)
  def toLong(a: Complex[A]): Long = algebra.toLong(a.real)
  def toFloat(a: Complex[A]): Float = algebra.toFloat(a.real)
  // def toDouble(a: Complex[A]): Double = algebra.toDouble(a.real)
  // def toBigInt(a: Complex[A]): BigInt = algebra.toBigInt(a.real)
  def toBigDecimal(a: Complex[A]): BigDecimal = algebra.toBigDecimal(a.real)
  // def toRational(a: Complex[A]): Rational = algebra.toRational(a.real)
  // def toAlgebraic(a: Complex[A]): Algebraic = algebra.toAlgebraic(a.real)
  // def toReal(a: Complex[A]): Real = algebra.toReal(a.real)
  def toNumber(a: Complex[A]): Number = algebra.toNumber(a.real)

  def toType[B: ConvertableTo](a: Complex[A]): B = sys.error("fixme")
  def toString(a: Complex[A]): String = a.toString
}

private[types] trait ConvertableFromByte extends ConvertableFrom[Byte] {
  def toByte(a: Byte): Byte = a
  def toShort(a: Byte): Short = a.toShort
  def toInt(a: Byte): Int = a.toInt
  def toLong(a: Byte): Long = a.toLong
  def toFloat(a: Byte): Float = a.toFloat
  // def toDouble(a: Byte): Double = a.toDouble
  // def toBigInt(a: Byte): BigInt = BigInt(a)
  def toBigDecimal(a: Byte): BigDecimal = BigDecimal(a)
  // def toRational(a: Byte): Rational = Rational(a)
  // def toAlgebraic(a: Byte): Algebraic = Algebraic(a)
  // def toReal(a: Byte): Real = Real(a)
  def toNumber(a: Byte): Number = Number(a)

  def toType[B: ConvertableTo](a: Byte): B = ConvertableTo[B].fromByte(a)
  def toString(a: Byte): String = a.toString
}

private[types] trait ConvertableFromShort extends ConvertableFrom[Short] {
  def toByte(a: Short): Byte = a.toByte
  def toShort(a: Short): Short = a
  def toInt(a: Short): Int = a.toInt
  def toLong(a: Short): Long = a.toLong
  def toFloat(a: Short): Float = a.toFloat
  // def toDouble(a: Short): Double = a.toDouble
  // def toBigInt(a: Short): BigInt = BigInt(a)
  def toBigDecimal(a: Short): BigDecimal = BigDecimal(a)
  // def toRational(a: Short): Rational = Rational(a)
  // def toAlgebraic(a: Short): Algebraic = Algebraic(a)
  // def toReal(a: Short): Real = Real(a)
  def toNumber(a: Short): Number = Number(a)

  def toType[B: ConvertableTo](a: Short): B = ConvertableTo[B].fromShort(a)
  def toString(a: Short): String = a.toString
}

private[types] trait ConvertableFromInt extends ConvertableFrom[Int] {
  def toByte(a: Int): Byte = a.toByte
  def toShort(a: Int): Short = a.toShort
  def toInt(a: Int): Int = a
  def toLong(a: Int): Long = a.toLong
  def toFloat(a: Int): Float = a.toFloat
  // def toDouble(a: Int): Double = a.toDouble
  // def toBigInt(a: Int): BigInt = BigInt(a)
  def toBigDecimal(a: Int): BigDecimal = BigDecimal(a)
  // def toRational(a: Int): Rational = Rational(a)
  // def toAlgebraic(a: Int): Algebraic = Algebraic(a)
  // def toReal(a: Int): Real = Real(a)
  def toNumber(a: Int): Number = Number(a)

  def toType[B: ConvertableTo](a: Int): B = ConvertableTo[B].fromInt(a)
  def toString(a: Int): String = a.toString
}

private[types] trait ConvertableFromLong extends ConvertableFrom[Long] {
  def toByte(a: Long): Byte = a.toByte
  def toShort(a: Long): Short = a.toShort
  def toInt(a: Long): Int = a.toInt
  def toLong(a: Long): Long = a
  def toFloat(a: Long): Float = a.toFloat
  // def toDouble(a: Long): Double = a.toDouble
  // def toBigInt(a: Long): BigInt = BigInt(a)
  def toBigDecimal(a: Long): BigDecimal = BigDecimal(a)
  // def toRational(a: Long): Rational = Rational(a)
  // def toAlgebraic(a: Long): Algebraic = Algebraic(a)
  // def toReal(a: Long): Real = Real(a)
  def toNumber(a: Long): Number = Number(a)

  def toType[B: ConvertableTo](a: Long): B = ConvertableTo[B].fromLong(a)
  def toString(a: Long): String = a.toString
}

private[types] trait ConvertableFromUByte extends ConvertableFrom[UByte] {
  def toByte(a: UByte): Byte = a.toByte
  def toShort(a: UByte): Short = a.toShort
  def toInt(a: UByte): Int = a.toInt
  def toLong(a: UByte): Long = a.toLong
  def toFloat(a: UByte): Float = a.toFloat
  // def toDouble(a: UByte): Double = a.toDouble
  // def toBigInt(a: UByte): BigInt = BigInt(a.toByte)
  def toBigDecimal(a: UByte): BigDecimal = BigDecimal(a.toByte)
  // def toRational(a: UByte): Rational = Rational(a.toByte)
  // def toAlgebraic(a: UByte): Algebraic = Algebraic(a.toByte)
  // def toReal(a: UByte): Real = Real(a.toByte)
  def toNumber(a: UByte): Number = Number(a.toByte)

  def toType[B: ConvertableTo](a: UByte): B = ConvertableTo[B].fromByte(a.toByte)
  def toString(a: UByte): String = a.toString
}

private[types] trait ConvertableFromUShort extends ConvertableFrom[UShort] {
  def toByte(a: UShort): Byte = a.toByte
  def toShort(a: UShort): Short = a.toShort
  def toInt(a: UShort): Int = a.toInt
  def toLong(a: UShort): Long = a.toLong
  def toFloat(a: UShort): Float = a.toFloat
  // def toDouble(a: UShort): Double = a.toDouble
  // def toBigInt(a: UShort): BigInt = BigInt(a.toByte)
  def toBigDecimal(a: UShort): BigDecimal = BigDecimal(a.toShort)
  // def toRational(a: UShort): Rational = Rational(a.toShort)
  // def toAlgebraic(a: UShort): Algebraic = Algebraic(a.toShort)
  // def toReal(a: UShort): Real = Real(a.toShort)
  def toNumber(a: UShort): Number = Number(a.toShort)

  def toType[B: ConvertableTo](a: UShort): B = ConvertableTo[B].fromShort(a.toShort)
  def toString(a: UShort): String = a.toString
}
