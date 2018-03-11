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

import spire.math.{UByte, UShort}

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait SupportedType[T] {
  @inline def dataType: DataType

  @throws[UnsupportedOperationException]
  @inline def cast[R: SupportedType](value: R, dataType: DataType): T = {
    throw InvalidDataTypeException("The Scala type of this data type is not supported.")
  }
}

object SupportedType {
  class SupportedTypeOps[T](val value: T) {
    @inline def dataType(implicit evidence: SupportedType[T]): DataType = evidence.dataType

    @inline def cast(dataType: DataType)
        (implicit evidence: SupportedType[T],
            dataTypeEvidence: SupportedType[dataType.ScalaType]): dataType.ScalaType = {
      dataTypeEvidence.cast(value, dataType)
    }
  }

  @inline final def apply[T](implicit evidence: SupportedType[T]): SupportedType[T] = evidence

  implicit def toSupportedTypeOps[T](value: T): SupportedTypeOps[T] = {
    new SupportedTypeOps(value)
  }

  implicit val stringIsSupportedType: SupportedType[String] = new SupportedType[String] {
    @inline override def dataType: DataType = STRING
    @inline override def cast[R: SupportedType](value: R, dataType: DataType): String = value.toString
  }

  implicit val booleanIsSupportedType: SupportedType[Boolean] = new SupportedType[Boolean] {
    @inline override def dataType: DataType = BOOLEAN
    @inline override def cast[R: SupportedType](value: R, dataType: DataType): Boolean = value match {
      case value: Boolean => value
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a boolean.")
    }
  }

  implicit val floatIsSupportedType: SupportedType[Float] = new SupportedType[Float] {
    @inline override def dataType: DataType = FLOAT32
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

  implicit val doubleIsSupportedType: SupportedType[Double] = new SupportedType[Double] {
    @inline override def dataType: DataType = FLOAT64
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

  implicit val byteIsSupportedType: SupportedType[Byte] = new SupportedType[Byte] {
    @inline override def dataType: DataType = INT8
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

  implicit val shortIsSupportedType: SupportedType[Short] = new SupportedType[Short] {
    @inline override def dataType: DataType = INT16
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

  implicit val intIsSupportedType: SupportedType[Int] = new SupportedType[Int] {
    @inline override def dataType: DataType = INT32
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

  implicit val longIsSupportedType: SupportedType[Long] = new SupportedType[Long] {
    @inline override def dataType: DataType = INT64
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

  implicit val uByteIsSupportedType: SupportedType[UByte] = new SupportedType[UByte] {
    @inline override def dataType: DataType = UINT8
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

  implicit val uShortIsSupportedType: SupportedType[UShort] = new SupportedType[UShort] {
    @inline override def dataType: DataType = UINT16
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
}
