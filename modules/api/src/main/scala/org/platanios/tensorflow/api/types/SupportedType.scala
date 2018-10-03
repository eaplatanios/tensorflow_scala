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

package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait SupportedType[+T] {
  @inline def dataType: DataType[T]

  @throws[InvalidDataTypeException]
  @inline def cast[R: SupportedType](value: R): T = {
    throw InvalidDataTypeException("The Scala type of this data type is not supported.")
  }
}

object SupportedType {
  implicit class SupportedTypeOps[T](val value: T)(implicit ev: SupportedType[T]) {
    @inline def dataType: DataType[T] = ev.dataType
    @inline def cast[R: SupportedType](dataType: DataType[R]): R = {
      implicitly[SupportedType[R]].cast(value)
    }
  }

  // TODO: [TYPES] Complete this.

  implicit val stringIsSupported: SupportedType[String] = new SupportedType[String] {
    @inline override def dataType: DataType[String] = STRING
    @inline override def cast[R: SupportedType](value: R): String = value.toString
  }

  implicit val booleanIsSupported: SupportedType[Boolean] = new SupportedType[Boolean] {
    @inline override def dataType: DataType[Boolean] = BOOLEAN
    @inline override def cast[R: SupportedType](value: R): Boolean = value match {
      case value: Boolean => value
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a boolean.")
    }
  }

  implicit val halfIsSupported: SupportedType[Half] = new SupportedType[Half] {
    @inline override def dataType: DataType[Half] = FLOAT16
    @inline override def cast[R: SupportedType](value: R): Half = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a half.")
    }
  }

  implicit val floatIsSupported: SupportedType[Float] = new SupportedType[Float] {
    @inline override def dataType: DataType[Float] = FLOAT32
    @inline override def cast[R: SupportedType](value: R): Float = value match {
      case value: Boolean => if (value) 1.0f else 0.0f
      case value: Float => value.toFloat
      case value: Double => value.toFloat
      case value: Byte => value.toFloat
      case value: Short => value.toFloat
      case value: Int => value.toFloat
      case value: Long => value.toFloat
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a float.")
    }
  }

  implicit val doubleIsSupported: SupportedType[Double] = new SupportedType[Double] {
    @inline override def dataType: DataType[Double] = FLOAT64
    @inline override def cast[R: SupportedType](value: R): Double = value match {
      case value: Boolean => if (value) 1.0 else 0.0
      case value: Float => value.toDouble
      case value: Double => value.toDouble
      case value: Byte => value.toDouble
      case value: Short => value.toDouble
      case value: Int => value.toDouble
      case value: Long => value.toDouble
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a double.")
    }
  }

  implicit val truncatedHalfIsSupported: SupportedType[TruncatedHalf] = new SupportedType[TruncatedHalf] {
    @inline override def dataType: DataType[TruncatedHalf] = BFLOAT16
    @inline override def cast[R: SupportedType](value: R): TruncatedHalf = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a truncated half.")
    }
  }

  implicit val complexFloatIsSupported: SupportedType[ComplexFloat] = new SupportedType[ComplexFloat] {
    @inline override def dataType: DataType[ComplexFloat] = COMPLEX64
    @inline override def cast[R: SupportedType](value: R): ComplexFloat = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a complex float.")
    }
  }

  implicit val complexDoubleIsSupported: SupportedType[ComplexDouble] = new SupportedType[ComplexDouble] {
    @inline override def dataType: DataType[ComplexDouble] = COMPLEX128
    @inline override def cast[R: SupportedType](value: R): ComplexDouble = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a complex double.")
    }
  }

  implicit val byteIsSupported: SupportedType[Byte] = new SupportedType[Byte] {
    @inline override def dataType: DataType[Byte] = INT8
    @inline override def cast[R: SupportedType](value: R): Byte = value match {
      case value: Boolean => if (value) 1 else 0
      case value: Float => value.toByte
      case value: Double => value.toByte
      case value: Byte => value.toByte
      case value: Short => value.toByte
      case value: Int => value.toByte
      case value: Long => value.toByte
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a byte.")
    }
  }

  implicit val shortIsSupported: SupportedType[Short] = new SupportedType[Short] {
    @inline override def dataType: DataType[Short] = INT16
    @inline override def cast[R: SupportedType](value: R): Short = value match {
      case value: Boolean => if (value) 1 else 0
      case value: Float => value.toShort
      case value: Double => value.toShort
      case value: Byte => value.toShort
      case value: Short => value.toShort
      case value: Int => value.toShort
      case value: Long => value.toShort
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a short.")
    }
  }

  implicit val intIsSupported: SupportedType[Int] = new SupportedType[Int] {
    @inline override def dataType: DataType[Int] = INT32
    @inline override def cast[R: SupportedType](value: R): Int = value match {
      case value: Boolean => if (value) 1 else 0
      case value: Float => value.toInt
      case value: Double => value.toInt
      case value: Byte => value.toInt
      case value: Short => value.toInt
      case value: Int => value.toInt
      case value: Long => value.toInt
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an integer.")
    }
  }

  implicit val longIsSupported: SupportedType[Long] = new SupportedType[Long] {
    @inline override def dataType: DataType[Long] = INT64
    @inline override def cast[R: SupportedType](value: R): Long = value match {
      case value: Boolean => if (value) 1L else 0L
      case value: Float => value.toLong
      case value: Double => value.toLong
      case value: Byte => value.toLong
      case value: Short => value.toLong
      case value: Int => value.toLong
      case value: Long => value.toLong
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a long.")
    }
  }

  implicit val uByteIsSupported: SupportedType[UByte] = new SupportedType[UByte] {
    @inline override def dataType: DataType[UByte] = UINT8
    @inline override def cast[R: SupportedType](value: R): UByte = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned byte.")
    }
  }

  implicit val uShortIsSupported: SupportedType[UShort] = new SupportedType[UShort] {
    @inline override def dataType: DataType[UShort] = UINT16
    @inline override def cast[R: SupportedType](value: R): UShort = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned short.")
    }
  }

  implicit val uIntIsSupported: SupportedType[UInt] = new SupportedType[UInt] {
    @inline override def dataType: DataType[UInt] = UINT32
    @inline override def cast[R: SupportedType](value: R): UInt = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned integer.")
    }
  }

  implicit val uLongIsSupported: SupportedType[ULong] = new SupportedType[ULong] {
    @inline override def dataType: DataType[ULong] = UINT64
    @inline override def cast[R: SupportedType](value: R): ULong = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned long.")
    }
  }

  implicit val qByteIsSupported: SupportedType[QByte] = new SupportedType[QByte] {
    @inline override def dataType: DataType[QByte] = QINT8
    @inline override def cast[R: SupportedType](value: R): QByte = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a quantized byte.")
    }
  }

  implicit val qShortIsSupported: SupportedType[QShort] = new SupportedType[QShort] {
    @inline override def dataType: DataType[QShort] = QINT16
    @inline override def cast[R: SupportedType](value: R): QShort = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a quantized short.")
    }
  }

  implicit val qIntIsSupported: SupportedType[QInt] = new SupportedType[QInt] {
    @inline override def dataType: DataType[QInt] = QINT32
    @inline override def cast[R: SupportedType](value: R): QInt = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a quantized integer.")
    }
  }

  implicit val qUByteIsSupported: SupportedType[QUByte] = new SupportedType[QUByte] {
    @inline override def dataType: DataType[QUByte] = QUINT8
    @inline override def cast[R: SupportedType](value: R): QUByte = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned quantized byte.")
    }
  }

  implicit val qUShortIsSupported: SupportedType[QUShort] = new SupportedType[QUShort] {
    @inline override def dataType: DataType[QUShort] = QUINT16
    @inline override def cast[R: SupportedType](value: R): QUShort = value match {
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned quantized short.")
    }
  }
}
